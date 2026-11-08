import subprocess

print("🧹 Cleaning up old processes, ports, and temp files...")

# 1. Kill everything running on ports 8000, 8001, and 11434
subprocess.run(["fuser", "-k", "8000/tcp"], capture_output=True)
subprocess.run(["fuser", "-k", "8001/tcp"], capture_output=True)
subprocess.run(["fuser", "-k", "11434/tcp"], capture_output=True)

# 2. Terminate any stray python/vllm/ollama or cloudflared processes
subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
subprocess.run(["pkill", "-9", "-f", "ollama"], capture_output=True)
subprocess.run(["pkill", "-9", "-f", "cloudflared"], capture_output=True)

# 3. Remove old log files
subprocess.run(["rm", "-f", "vllm_output.log", "vllm.log", "ollama.log"], capture_output=True)

print("✨ All clean! Environment is spotless.")