# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in TorchWM, please report it privately by opening a security advisory on GitHub:

https://github.com/paramthakkar123/torchwm/security/advisories/new

Please do **not** report security vulnerabilities through public GitHub issues, discussions, or pull requests.

## What to Include

- A clear description of the vulnerability
- Steps to reproduce (if applicable)
- Affected versions
- Any potential mitigations you have identified

## Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 5 business days
- **Fix and release**: timeline depends on severity, typically 7–30 days

## Scope

TorchWM uses `torch.load(weights_only=True)` by default and `subprocess.run(shell=False)` throughout the codebase. However, if you find any code path that deviates from these patterns, please report it.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |
| 0.4.x   | Yes       |
| < 0.4   | No        |
