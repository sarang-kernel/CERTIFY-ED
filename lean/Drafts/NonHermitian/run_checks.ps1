# Drafts/NonHermitian/run_checks.ps1
# Build, axioms, text search, leanchecker and nanoda for CertifyMathlib.NonHermitian, one step after the other.
# Run from any directory: powershell -File Drafts/NonHermitian/run_checks.ps1 [-Skip build,axioms,...] [-Python <python.exe with psutil>]
param([string[]]$Skip = @(), [string]$Python = "")
$r = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$d = Join-Path $r "Drafts\NonHermitian"
$envj = Get-Content -Raw (Join-Path $r "logs\lean_env.json") | ConvertFrom-Json
$py = $Python
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
foreach ($p in $envj.PSObject.Properties) { Set-Item -Path ("Env:" + $p.Name) -Value $p.Value }
Set-Location $r
$utf8 = New-Object System.Text.UTF8Encoding($false)
$summary = Join-Path $d "run_checks_summary.txt"
if ($Skip.Count -eq 0) { Set-Content -Path $summary -Value ("run started " + (Get-Date -Format "yyyy-MM-dd HH:mm")) }
function Step($name, [scriptblock]$body) {
  if ($Skip -contains $name) { return }
  $t0 = Get-Date
  $out = & $body 2>&1 | Out-String
  $code = $LASTEXITCODE
  $dt = [int]((Get-Date) - $t0).TotalSeconds
  [System.IO.File]::WriteAllText((Join-Path $d ("check_" + $name + ".log")), $out, $utf8)
  Add-Content -Path $summary -Value ("{0} exit={1} seconds={2} date={3}" -f $name, $code, $dt, (Get-Date -Format "yyyy-MM-dd HH:mm"))
}
Step "build" { lake build CertifyMathlib.NonHermitian }
Step "axioms" { lake env lean Drafts/NonHermitian/PrintAxioms.lean }
Step "textsearch" {
  $files = @(Get-ChildItem -Path (Join-Path $r "CertifyMathlib\NonHermitian") -Filter *.lean) + @(Get-Item (Join-Path $r "CertifyMathlib\NonHermitian.lean")) + @(Get-ChildItem -Path $d -Filter *.lean)
  foreach ($pat in @("sorry", "admit", "native_decide", "^\s*axiom\s", "set_option")) {
    $n = 0
    foreach ($f in $files) { $n += @(Select-String -Path $f.FullName -Pattern $pat).Count }
    "{0} {1}" -f $pat, $n
  }
  "files {0}" -f $files.Count
}
Step "leanchecker" { lake env leanchecker CertifyMathlib.NonHermitian.BauerFike CertifyMathlib.NonHermitian }
Step "nanoda" {
  & $py Drafts/NonHermitian/../Counting/nanoda_counting.py --lean-bin $envj.PATH.Split(';')[0] --work (Join-Path $d "nanoda_work") --theorems (Join-Path $d "nanoda_theorems.txt") --out (Join-Path $d "nanoda_results.jsonl") --mem-mib 4096 --timeout 1800
}
