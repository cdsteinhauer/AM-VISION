param(
    [string]$TargetHost = "192.168.20.241",
    [string]$User = "csteinhauer",
    [int]$Port = 22,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_robot_vision"
)

$ErrorActionPreference = "Stop"

$sshDir = Split-Path -Parent $KeyPath
if (-not (Test-Path -LiteralPath $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir | Out-Null
}

if (-not (Test-Path -LiteralPath $KeyPath)) {
    ssh-keygen -t ed25519 -N "" -f $KeyPath -C "robot_vision_deploy"
}

$publicKey = Get-Content -Raw "$KeyPath.pub"
$escapedKey = $publicKey.Replace("'", "'\''").Trim()

ssh -p $Port -o StrictHostKeyChecking=accept-new "$User@$TargetHost" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && grep -qxF '$escapedKey' ~/.ssh/authorized_keys || echo '$escapedKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

ssh -i $KeyPath -p $Port -o BatchMode=yes "$User@$TargetHost" "echo key-auth-ok"

Write-Host "SSH key auth ready for $User@$TargetHost using $KeyPath"
