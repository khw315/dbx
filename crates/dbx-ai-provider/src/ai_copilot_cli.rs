use crate::agent_events::AgentEvent;
use crate::ai::{AiConfig, AiEffortCapability, AiModelInfo, AiTestConnectionResult};
use crate::ai_cli_agent::{
    build_cli_agent_prompt, cli_command, dbx_mcp_enabled_tools, dbx_mcp_scope_env, model_infos, parse_cli_jsonl_event,
    run_cli_jsonl_agent, CliAgentCommandSpec, CliAgentJsonlDialect, CliAgentProcessSpec, CliAgentRunOptions,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

const COPILOT_COMMAND_TIMEOUT: Duration = Duration::from_secs(10);
const COPILOT_CONTROL_ENV: &[&str] = &["COPILOT_CONFIG_DIR", "COPILOT_DATA_DIR"];

pub type CopilotRunOptions = CliAgentRunOptions;
pub type CopilotCommandSpec = CliAgentCommandSpec;

pub const DEFAULT_COPILOT_MODELS: &[&str] = &["default", "gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "o1", "o3-mini"];

struct CopilotIsolatedRuntime {
    root: PathBuf,
    workspace: PathBuf,
    config: PathBuf,
    data: PathBuf,
}

impl CopilotIsolatedRuntime {
    fn create(options: Option<&CopilotRunOptions>) -> Result<Self, String> {
        let root = env::temp_dir().join(format!("dbx-copilot-{}", uuid::Uuid::new_v4()));
        let workspace = root.join("workspace");
        let copilot_dir = workspace.join(".copilot");
        let config = root.join("config");
        let data = root.join("data");
        for path in [&workspace, &copilot_dir, &config, &data] {
            std::fs::create_dir_all(path)
                .map_err(|error| format!("[copilotRunFailed] Failed to create isolated Copilot directory: {error}"))?;
        }

        if let Some(options) = options {
            Self::write_mcp_config(&copilot_dir, options)?;
        }

        Ok(Self { root, workspace, config, data })
    }

    fn write_mcp_config(copilot_dir: &Path, options: &CopilotRunOptions) -> Result<(), String> {
        let command = options
            .mcp_server_command
            .as_ref()
            .cloned()
            .unwrap_or_else(|| CopilotCommandSpec { program: "dbx-mcp-server".to_string(), args: Vec::new() });
        let env = dbx_mcp_scope_env(options).into_iter().collect::<BTreeMap<_, _>>();
        let config = json!({
            "mcpServers": {
                "dbx": {
                    "command": command.program,
                    "args": command.args,
                    "env": env
                }
            }
        });
        write_json_file(&copilot_dir.join("mcp.json"), &config, "Copilot MCP")
    }

    fn process_env(&self, config: &AiConfig) -> Result<Vec<(String, String)>, String> {
        let mut values = BTreeMap::from_iter(copilot_cli_env(config)?);
        values.insert("COPILOT_CONFIG_DIR".to_string(), self.config.to_string_lossy().to_string());
        values.insert("COPILOT_DATA_DIR".to_string(), self.data.to_string_lossy().to_string());
        Ok(values.into_iter().collect())
    }
}

impl Drop for CopilotIsolatedRuntime {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn write_json_file(path: &Path, value: &Value, label: &str) -> Result<(), String> {
    let content = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("[copilotRunFailed] Failed to serialize {label} configuration: {error}"))?;
    std::fs::write(path, content)
        .map_err(|error| format!("[copilotRunFailed] Failed to write {label} configuration: {error}"))
}

fn copilot_program(config: &AiConfig) -> String {
    config.copilot_cli_path.as_deref().map(str::trim).filter(|value| !value.is_empty()).unwrap_or("copilot").to_string()
}

pub fn resolve_copilot_command(config: &AiConfig) -> Result<CopilotCommandSpec, String> {
    let raw = copilot_program(config);
    if raw.contains('=') {
        return Err("[copilotCliPathInvalid] Copilot CLI path should contain only the executable path. Add environment variables in the Copilot CLI environment variables section.".to_string());
    }
    let parts = shlex::split(&raw).unwrap_or_default();
    let Some((program, args)) = parts.split_first() else {
        return Err("[copilotCliPathInvalid] Copilot CLI executable path cannot be empty.".to_string());
    };
    let program_path = Path::new(program);
    if program_path.is_dir() {
        return Err(
            "[copilotCliPathInvalid] Copilot CLI path should point to the copilot executable or a directory containing copilot."
                .to_string(),
        );
    }
    if (program.contains('/') || program.contains('\\')) && !program_path.exists() {
        return Err("[copilotCliPathInvalid] Copilot CLI executable does not exist.".to_string());
    }
    Ok(CopilotCommandSpec { program: program.clone(), args: args.to_vec() })
}

pub fn copilot_cli_env(config: &AiConfig) -> Result<Vec<(String, String)>, String> {
    let mut env = Vec::new();
    for (key, value) in &config.copilot_cli_env {
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        let upper = key.to_ascii_uppercase();
        if upper.starts_with("DBX_MCP_") || COPILOT_CONTROL_ENV.contains(&upper.as_str()) {
            return Err(format!("[copilotCliEnvReserved] Reserved environment variable: {key}"));
        }
        env.push((key.to_string(), value.clone()));
    }
    Ok(env)
}

pub fn build_copilot_prompt(system_prompt: &str, messages: &[crate::ai::AiMessage], allow_write_sql: bool) -> String {
    build_cli_agent_prompt("GitHub Copilot", system_prompt, messages, allow_write_sql)
}

pub async fn list_copilot_models(config: &AiConfig) -> Result<Vec<AiModelInfo>, String> {
    let mut infos = model_infos(DEFAULT_COPILOT_MODELS);
    for info in &mut infos {
        info.effort_capability = Some(AiEffortCapability::Unsupported);
    }
    Ok(infos)
}

pub async fn test_copilot_connection(config: &AiConfig) -> Result<AiTestConnectionResult, String> {
    let start = Instant::now();
    let runtime = CopilotIsolatedRuntime::create(None)?;
    let command = resolve_copilot_command(config)?;
    let mut process = cli_command(&command.program);
    process.args(&command.args).arg("--version");
    for key in COPILOT_CONTROL_ENV {
        process.env_remove(key);
    }
    process.envs(runtime.process_env(config)?).current_dir(&runtime.workspace).kill_on_drop(true);
    let output = tokio::time::timeout(COPILOT_COMMAND_TIMEOUT, process.output())
        .await
        .map_err(|_| "[copilotTimeout] Copilot CLI version check timed out".to_string())?
        .map_err(|error| classify_copilot_spawn_error(&error.to_string()))?;
    if !output.status.success() {
        return Err(classify_copilot_run_error(&combined_output(&output.stderr, &output.stdout)));
    }
    let elapsed = start.elapsed();
    let version_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let message = if version_str.is_empty() {
        format!("OK - {}ms", elapsed.as_millis())
    } else {
        format!("{version_str} - {}ms", elapsed.as_millis())
    };
    Ok(AiTestConnectionResult {
        success: true,
        message,
        latency_ms: Some(elapsed.as_millis() as u64),
        model_used: config.model.trim().to_string(),
        error_category: None,
    })
}

fn combined_output(stderr: &[u8], stdout: &[u8]) -> String {
    let stderr = String::from_utf8_lossy(stderr);
    let stdout = String::from_utf8_lossy(stdout);
    [stderr.trim(), stdout.trim()].into_iter().filter(|part| !part.is_empty()).collect::<Vec<_>>().join("\n")
}

fn classify_copilot_spawn_error(message: &str) -> String {
    let lower = message.to_ascii_lowercase();
    if lower.contains("no such file") || lower.contains("not found") || lower.contains("cannot find") {
        format!("[copilotNotInstalled] {message}")
    } else {
        format!("[copilotRunFailed] {message}")
    }
}

fn classify_copilot_run_error(message: &str) -> String {
    if message.starts_with("[copilot") || message.starts_with("[dbxMcpMissing]") {
        return message.to_string();
    }
    let lower = message.to_ascii_lowercase();
    if lower.contains("not authenticated")
        || lower.contains("authentication required")
        || lower.contains("unauthorized")
        || lower.contains("please login")
        || lower.contains("please sign in")
        || lower.contains("copilot auth")
    {
        format!("[copilotNotAuthenticated] {message}")
    } else if lower.contains("dbx-mcp-server") || lower.contains("enoent") {
        format!("[dbxMcpMissing] {message}")
    } else if lower.contains("mcp") && (lower.contains("dbx") || lower.contains("server")) {
        format!("[copilotMcpStartupFailed] {message}")
    } else if lower.contains("json") || lower.contains("protocol") {
        format!("[copilotProtocolError] {message}")
    } else {
        format!("[copilotRunFailed] {message}")
    }
}

pub fn parse_copilot_jsonl_event(line: &str) -> Option<Vec<AgentEvent>> {
    parse_cli_jsonl_event(line, CliAgentJsonlDialect::CursorPrint)
}

pub async fn run_copilot_agent(
    config: &AiConfig,
    prompt: &str,
    options: CopilotRunOptions,
    cancelled: &Notify,
    on_event: impl Fn(AgentEvent) + Send + Sync + 'static,
) -> Result<String, String> {
    let runtime = CopilotIsolatedRuntime::create(Some(&options))?;
    let resolved = resolve_copilot_command(config)?;
    let mut command = CopilotCommandSpec { program: resolved.program, args: Vec::new() };
    command.args.splice(0..0, resolved.args);
    command.args.extend(["-p".to_string(), prompt.to_string()]);

    let result = run_cli_jsonl_agent(
        CliAgentProcessSpec {
            command,
            env: runtime.process_env(config)?,
            env_remove: COPILOT_CONTROL_ENV.iter().map(|value| (*value).to_string()).collect(),
            current_dir: Some(runtime.workspace.clone()),
            stdin: None,
            dialect: CliAgentJsonlDialect::CursorPrint,
            classify_spawn_error: classify_copilot_spawn_error,
            classify_run_error: classify_copilot_run_error,
        },
        cancelled,
        on_event,
    )
    .await;
    result
}
