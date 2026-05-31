#Requires -Version 5
<#
.SYNOPSIS
    Set up agentic-inferno on Windows.

.DESCRIPTION
    Installs Rust (if missing), ensures a C linker is available, builds the
    release binary, and walks you through entering API keys into a .env file.
    Safe to re-run.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1
#>

$ErrorActionPreference = 'Stop'

# Run from the repository root (the directory holding this script).
Set-Location -Path $PSScriptRoot

$Keys = @('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'MOONSHOT_API_KEY')

function Write-Info { param([string]$Message) Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Write-Warn { param([string]$Message) Write-Warning $Message }

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# --- 1. Ensure Rust -------------------------------------------------------

function Install-Rust {
    if ((Test-Command 'cargo') -and (Test-Command 'rustc')) {
        Write-Info "Rust found: $(rustc --version)"
        return
    }

    Write-Info 'Rust not found. Installing via rustup...'

    if (Test-Command 'winget') {
        try {
            winget install --id Rustlang.Rustup -e --accept-source-agreements --accept-package-agreements
        }
        catch {
            Write-Warn "winget install of Rustup failed: $($_.Exception.Message). Falling back to rustup-init.exe."
        }
    }

    if (-not (Test-Command 'cargo')) {
        # Fallback: download and run rustup-init.exe directly.
        $rustupInit = Join-Path $env:TEMP 'rustup-init.exe'
        Write-Info 'Downloading rustup-init.exe...'
        Invoke-WebRequest -Uri 'https://win.rustup.rs/x86_64' -OutFile $rustupInit
        & $rustupInit -y
    }

    # Make cargo available in the current session for the rest of this run.
    $cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
    if (Test-Path $cargoBin) {
        $env:PATH = "$cargoBin;$env:PATH"
    }

    if (-not (Test-Command 'cargo')) {
        Write-Warn 'cargo is still not on PATH. Open a new terminal (so PATH refreshes) and re-run this script.'
        exit 1
    }

    rustup default stable | Out-Null
    Write-Info "Rust installed: $(rustc --version)"
}

# --- 2. Ensure a C linker -------------------------------------------------

function Install-CLinker {
    # The MSVC toolchain needs the C++ build tools (link.exe). Detect a
    # plausible install; if absent, try to install, otherwise offer the
    # GNU toolchain as a no-Visual-Studio fallback. Detection is best-effort
    # so we warn rather than hard-fail.
    $haveLink = Test-Command 'link'
    $haveVsWhere = Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    $haveMsvc = $false
    if ($haveVsWhere) {
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($found) { $haveMsvc = $true }
    }

    if ($haveLink -or $haveMsvc) {
        Write-Info 'C linker (MSVC build tools) detected.'
        return
    }

    Write-Warn 'MSVC C++ build tools were not detected. They provide the linker that the default Rust toolchain needs.'

    if (Test-Command 'winget') {
        Write-Info 'Attempting to install Visual Studio 2022 Build Tools (C++ workload)...'
        try {
            winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
                --accept-source-agreements --accept-package-agreements `
                --override '--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
            Write-Info 'Build Tools install attempted. A reboot or new terminal may be required.'
            return
        }
        catch {
            Write-Warn "Automatic Build Tools install failed: $($_.Exception.Message)"
        }
    }

    Write-Warn @'
Could not install the MSVC build tools automatically. You have two options:

  1. Install "Build Tools for Visual Studio 2022" and select the
     "Desktop development with C++" workload, then re-run this script:
     https://visualstudio.microsoft.com/visual-cpp-build-tools/

  2. Use the GNU toolchain, which needs no Visual Studio:
     rustup toolchain install stable-x86_64-pc-windows-gnu
     rustup default stable-x86_64-pc-windows-gnu

Continuing — the build will fail if no linker is present.
'@
}

# --- 3. Build -------------------------------------------------------------

function Build-Release {
    Write-Info 'Building release binary (cargo build --release)...'
    cargo build --release
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build --release failed (exit $LASTEXITCODE). See the linker guidance above if this is a link error."
    }
    Write-Info 'Build complete: .\target\release\agentic-inferno.exe'
}

# --- 4. API-key setup -----------------------------------------------------

# Return the current value for a key from .env, or '' if absent/placeholder.
function Get-EnvValue {
    param([string]$Key)
    if (-not (Test-Path '.env')) { return '' }
    foreach ($line in Get-Content '.env') {
        if ($line -like "$Key=*") {
            return $line.Substring($Key.Length + 1)
        }
    }
    return ''
}

# Replace (or append) the KEY=value line in .env with a literal value.
# Rebuilds the file line by line so the value is never reinterpreted
# (API keys can contain $ & and other regex-special characters).
function Set-EnvValue {
    param([string]$Key, [string]$Value)

    $lines = @()
    if (Test-Path '.env') {
        $lines = @(Get-Content '.env')
    }

    $out = New-Object System.Collections.Generic.List[string]
    $replaced = $false
    foreach ($line in $lines) {
        if ($line -like "$Key=*") {
            $out.Add("$Key=$Value")
            $replaced = $true
        }
        else {
            $out.Add($line)
        }
    }
    if (-not $replaced) {
        $out.Add("$Key=$Value")
    }

    Set-Content -Path '.env' -Value $out
}

# Read a hidden line of input and return it as plain text (or '' if blank).
function Read-SecretLine {
    param([string]$Prompt)
    $secure = Read-Host -Prompt $Prompt -AsSecureString
    if (-not $secure -or $secure.Length -eq 0) { return '' }
    return [System.Net.NetworkCredential]::new('', $secure).Password
}

function Set-ApiKeys {
    Write-Info 'Setting up API keys in .env'

    if (-not (Test-Path '.env')) {
        if (Test-Path '.env.example') {
            Copy-Item '.env.example' '.env'
            Write-Host 'Created .env from .env.example.'
        }
        else {
            New-Item -Path '.env' -ItemType File | Out-Null
            Write-Host 'Created an empty .env (.env.example not found).'
        }
    }

    Write-Host 'Enter each API key (input hidden). Leave blank to skip a provider.'
    Write-Host ''

    foreach ($key in $Keys) {
        $existing = Get-EnvValue -Key $key
        $hasReal = $existing -and -not ($existing -like 'sk-...*') -and -not ($existing -like 'sk-ant-...*')

        if ($hasReal) {
            $ans = Read-Host "$key already has a value. Overwrite? [y/N]"
            if ($ans -notmatch '^[Yy]$') {
                Write-Host "Keeping existing $key."
                continue
            }
        }

        $value = Read-SecretLine -Prompt $key
        if ([string]::IsNullOrEmpty($value)) {
            Write-Host "Skipped $key."
            continue
        }
        Set-EnvValue -Key $key -Value $value
        Write-Host "Saved $key."
    }

    Write-Info '.env updated.'
}

# --- 5. Optional: claude CLI check ---------------------------------------

function Test-ClaudeCli {
    if ((Test-Command 'claude.cmd') -or (Test-Command 'claude.exe') -or (Test-Command 'claude')) {
        Write-Info 'claude CLI found (needed only for Anthropic models).'
    }
    else {
        Write-Warn 'claude CLI not found. It is only required if you plan to use Anthropic models (claude-*, opus, sonnet, haiku). Install: https://docs.anthropic.com/en/docs/claude-code/overview'
    }
}

# --- Main -----------------------------------------------------------------

Install-Rust
Install-CLinker
Build-Release
Set-ApiKeys
Test-ClaudeCli

Write-Info 'Done.'
Write-Host 'Run it, for example:'
Write-Host '  .\target\release\agentic-inferno.exe --writer-model gpt-4o --input my-draft.md'
