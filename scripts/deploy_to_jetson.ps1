param(
    [string]$TargetHost = "jetson.local",
    [string]$FallbackHost = "192.168.20.241",
    [string]$User = "csteinhauer",
    [int]$Port = 22,
    [string]$RemoteAppDir = "/home/csteinhauer/robot_vision",
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519_robot_vision",
    [switch]$DryRun,
    [switch]$ColconBuild,
    [switch]$AllowPasswordPrompt
)

$ErrorActionPreference = "Stop"

$localRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$remotePackage = $RemoteAppDir
$excludeDirs = @(".git", "__pycache__", ".pytest_cache", ".venv", "build", "install", "log", "reports", "sample_output", "pytest-temp")
$excludeFiles = @("*.pyc", "*.pyo")
$sshOptions = @("-p", "$Port", "-o", "StrictHostKeyChecking=accept-new")
if (Test-Path -LiteralPath $IdentityFile) {
    $sshOptions += @("-i", $IdentityFile)
}

function Test-SshHost {
    param([string]$HostName)
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $probeOptions = $sshOptions + @("-o", "ConnectTimeout=5")
        if (-not $AllowPasswordPrompt) {
            $probeOptions += @("-o", "BatchMode=yes")
        }
        ssh @probeOptions "$User@$HostName" "echo ok" *> $null
        return $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
}

$hostToUse = $TargetHost
if (-not (Test-SshHost $TargetHost)) {
    Write-Host "Primary host $TargetHost did not respond; trying $FallbackHost"
    if (-not (Test-SshHost $FallbackHost)) {
        throw "Could not connect to $TargetHost or $FallbackHost over SSH."
    }
    $hostToUse = $FallbackHost
}

Write-Host "Using SSH target $User@$hostToUse"
if (-not $DryRun) {
    ssh @sshOptions "$User@$hostToUse" "mkdir -p '$remotePackage'"
}

$stageRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("robot_vision_deploy_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stageRoot | Out-Null

try {
    $robocopyArgs = @(
        $localRoot,
        $stageRoot,
        "/E",
        "/XD"
    ) + $excludeDirs + @("/XF") + $excludeFiles + @("/NFL", "/NDL", "/NJH", "/NJS", "/NP")

    & robocopy @robocopyArgs | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "Robocopy staging failed with exit code $LASTEXITCODE"
    }

    if ($DryRun) {
        Write-Host "DRY RUN staged files:"
        Get-ChildItem -Recurse -File $stageRoot | ForEach-Object {
            Write-Host $_.FullName.Replace($stageRoot, "")
        }
        Write-Host "Dry run complete. No files copied."
        exit 0
    }

    $scpOptions = @("-P", "$Port", "-o", "StrictHostKeyChecking=accept-new")
    if (Test-Path -LiteralPath $IdentityFile) {
        $scpOptions += @("-i", $IdentityFile)
    }
    scp @scpOptions -r (Join-Path $stageRoot "*") "${User}@${hostToUse}:$remotePackage/"
}
finally {
    if (Test-Path $stageRoot) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
}

ssh @sshOptions "$User@$hostToUse" "cd '$remotePackage' && export ROS_DOMAIN_ID=77 && python3 -m compileall -q robot_vision"
ssh @sshOptions "$User@$hostToUse" "cd '$remotePackage' && python3 -m pip install --user -r requirements.txt"

if ($ColconBuild) {
    ssh @sshOptions "$User@$hostToUse" "cd '$remotePackage' && export ROS_DOMAIN_ID=77 && colcon build --packages-select robot_vision"
}

Write-Host "Deploy complete: ${User}@${hostToUse}:$remotePackage"
