param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $Root ".venv\Scripts\streamlit.exe"

function Assert-VenvTool {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path $Path)) {
        throw "Missing $Name at '$Path'. Activate or create the project's virtual environment first."
    }
}

function Invoke-InDir {
    param(
        [string]$Path,
        [scriptblock]$Script
    )

    Push-Location $Path
    try {
        & $Script
    }
    finally {
        Pop-Location
    }
}

switch ($Target) {
    "install-ml" {
        Assert-VenvTool -Path $VenvPython -Name "Python"
        Invoke-InDir (Join-Path $Root "ml_pipeline") {
            & $VenvPython -m pip install -r requirements.txt
        }
    }
    "install-app" {
        Assert-VenvTool -Path $VenvPython -Name "Python"
        Invoke-InDir (Join-Path $Root "streamlit_app") {
            & $VenvPython -m pip install -r requirements.txt
        }
    }
    "install" {
        & $PSCommandPath "install-ml"
        & $PSCommandPath "install-app"
    }
    "train" {
        Assert-VenvTool -Path $VenvPython -Name "Python"
        Invoke-InDir (Join-Path $Root "ml_pipeline") {
            & $VenvPython train.py
        }
    }
    "train-large" {
        Assert-VenvTool -Path $VenvPython -Name "Python"
        Invoke-InDir (Join-Path $Root "ml_pipeline") {
            & $VenvPython train.py --rows 500000
        }
    }
    "train-force" {
        Assert-VenvTool -Path $VenvPython -Name "Python"
        Invoke-InDir (Join-Path $Root "ml_pipeline") {
            & $VenvPython train.py --force
        }
    }
    "train-skip-eda" {
        Assert-VenvTool -Path $VenvPython -Name "Python"
        Invoke-InDir (Join-Path $Root "ml_pipeline") {
            & $VenvPython train.py --skip-eda
        }
    }
    "app" {
        Assert-VenvTool -Path $VenvStreamlit -Name "Streamlit"
        Invoke-InDir (Join-Path $Root "streamlit_app") {
            & $VenvStreamlit run app.py
        }
    }
    "build" {
        docker-compose build
    }
    "up" {
        docker-compose up --build
    }
    "up-detached" {
        docker-compose up --build -d
    }
    "down" {
        docker-compose down
    }
    "logs" {
        docker-compose logs -f
    }
    "logs-ml" {
        docker-compose logs -f ml_pipeline
    }
    "logs-app" {
        docker-compose logs -f streamlit_app
    }
    "retrain" {
        docker-compose run --rm ml_pipeline python train.py --force
        docker-compose restart streamlit_app
    }
    "clean-models" {
        $paths = @(
            "ml_pipeline\models\model.pkl",
            "ml_pipeline\models\encoders.pkl",
            "ml_pipeline\models\features.pkl",
            "ml_pipeline\models\model_meta.json"
        )

        foreach ($relativePath in $paths) {
            $fullPath = Join-Path $Root $relativePath
            if (Test-Path $fullPath) {
                Remove-Item $fullPath -Force
            }
        }
    }
    "clean" {
        Get-ChildItem -Path $Root -Recurse -Filter *.pyc -File | Remove-Item -Force
        Get-ChildItem -Path $Root -Recurse -Directory | Where-Object {
            $_.Name -in @("__pycache__", ".pytest_cache")
        } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

        $patterns = @(
            "ml_pipeline\data\processed\*.csv",
            "ml_pipeline\reports\*.png",
            "ml_pipeline\reports\insights.json",
            "ml_pipeline\logs\*.log"
        )

        foreach ($pattern in $patterns) {
            Get-ChildItem -Path (Join-Path $Root $pattern) -ErrorAction SilentlyContinue | Remove-Item -Force
        }

        & $PSCommandPath "clean-models"
    }
    default {
        @"
AirFair Vista commands

PowerShell:
  .\tasks.ps1 install
  .\tasks.ps1 train
  .\tasks.ps1 app
  .\tasks.ps1 up

Supported targets:
  install, install-ml, install-app
  train, train-large, train-force, train-skip-eda
  app
  build, up, up-detached, down, logs, logs-ml, logs-app, retrain
  clean, clean-models
"@ | Write-Host
    }
}
