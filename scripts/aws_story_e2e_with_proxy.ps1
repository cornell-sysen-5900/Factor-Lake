param(
    [string]$BucketName = "sysen-5900-factor-lake",
    [string]$Region = "",
    [string]$AwsProfile = "Sysen5900Student",
    [string]$ProxyPathPart = "browser"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$e2eScript = Join-Path $PSScriptRoot "aws_story_e2e_test.ps1"

if (-not (Test-Path $e2eScript)) {
    throw "Base script not found: $e2eScript"
}

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    $candidate = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
    if (Test-Path $candidate) {
        $env:Path = "C:\Program Files\Amazon\AWSCLIV2;" + $env:Path
    }
}
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw "aws CLI not found. Install AWS CLI v2 and rerun."
}

$startTime = Get-Date

$e2eArgs = @{
    BucketName = $BucketName
    AwsProfile = $AwsProfile
}
if (-not [string]::IsNullOrWhiteSpace($Region)) {
    $e2eArgs.Region = $Region
}

Write-Host "Running base E2E script..."
& $e2eScript @e2eArgs

$evidenceRoot = Join-Path $repoRoot "evidence"
$latestRun = Get-ChildItem -Path $evidenceRoot -Directory |
    Where-Object { $_.Name -like "run-*" } |
    Sort-Object Name -Descending |
    Select-Object -First 1

if (-not $latestRun) {
    throw "No run-* evidence directory found after E2E script execution."
}

$summaryPath = Join-Path $latestRun.FullName "summary.json"
if (-not (Test-Path $summaryPath)) {
    throw "summary.json missing for run directory: $($latestRun.FullName)"
}

$summary = Get-Content -Path $summaryPath -Raw | ConvertFrom-Json
$restApiId = $summary.restApiId
$accountId = $summary.accountId
$effectiveRegion = $summary.region
$apiInvokeUrl = $summary.apiInvokeUrl
$apiKeyId = $summary.apiKeyId

if ([string]::IsNullOrWhiteSpace($restApiId) -or [string]::IsNullOrWhiteSpace($apiKeyId)) {
    throw "Cannot configure proxy because restApiId or apiKeyId is missing in summary.json"
}

if ([string]::IsNullOrWhiteSpace($effectiveRegion)) {
    $effectiveRegion = "us-east-1"
}

$invokeTrace = Join-Path $latestRun.FullName "_invoke-trace.log"

function Invoke-AwsStep {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$AwsArgs,
        [Parameter(Mandatory = $true)]
        [string]$OutputFile
    )

    $allArgs = @("--no-cli-pager", "--cli-connect-timeout", "15", "--cli-read-timeout", "60")
    if (-not [string]::IsNullOrWhiteSpace($AwsProfile)) {
        $allArgs += @("--profile", $AwsProfile)
    }
    if (-not [string]::IsNullOrWhiteSpace($effectiveRegion)) {
        $allArgs += @("--region", $effectiveRegion)
    }
    $allArgs += $AwsArgs

    $cmdLine = "aws " + ($allArgs -join " ")
    Add-Content -Path $invokeTrace -Value ((Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " START " + $cmdLine)

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = (& aws @allArgs 2>&1 | Out-String)
        $text = ([string]$raw).Trim()
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }

    @(
        "COMMAND: $cmdLine"
        "EXIT_CODE: $exitCode"
        "OUTPUT:"
        $text
    ) | Set-Content -Path $OutputFile

    Add-Content -Path $invokeTrace -Value ((Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " END exit=" + $exitCode + " " + $cmdLine)

    if ($exitCode -ne 0) {
        throw "AWS command failed: $cmdLine"
    }

    return $text
}

Write-Host "Configuring browser proxy endpoint /$ProxyPathPart ..."

$proxyConfigPath = Join-Path $latestRun.FullName "07a-proxy-config.txt"
@(
    "restApiId=$restApiId"
    "region=$effectiveRegion"
    "proxyPathPart=$ProxyPathPart"
    "targetUrl=$apiInvokeUrl"
) | Set-Content -Path $proxyConfigPath

$resourcesRaw = Invoke-AwsStep -AwsArgs @("apigateway", "get-resources", "--rest-api-id", $restApiId, "--output", "json") -OutputFile (Join-Path $latestRun.FullName "07b-proxy-get-resources.txt")
$resources = $resourcesRaw | ConvertFrom-Json
$rootId = ($resources.items | Where-Object { $_.path -eq "/" }).id
$proxyResource = $resources.items | Where-Object { $_.path -eq "/$ProxyPathPart" } | Select-Object -First 1

if ($proxyResource) {
    @(
        "COMMAND: reuse-existing-proxy-resource"
        "EXIT_CODE: 0"
        "OUTPUT:"
        "Reused resource /$ProxyPathPart with id $($proxyResource.id)"
    ) | Set-Content -Path (Join-Path $latestRun.FullName "07c-proxy-create-resource.txt")
    $proxyResourceId = $proxyResource.id
}
else {
    $proxyResourceRaw = Invoke-AwsStep -AwsArgs @("apigateway", "create-resource", "--rest-api-id", $restApiId, "--parent-id", $rootId, "--path-part", $ProxyPathPart, "--output", "json") -OutputFile (Join-Path $latestRun.FullName "07c-proxy-create-resource.txt")
    $proxyResourceId = ($proxyResourceRaw | ConvertFrom-Json).id
}

Invoke-AwsStep -AwsArgs @("apigateway", "put-method", "--rest-api-id", $restApiId, "--resource-id", $proxyResourceId, "--http-method", "GET", "--authorization-type", "NONE", "--output", "json") -OutputFile (Join-Path $latestRun.FullName "07d-proxy-put-method.txt") | Out-Null

$apiKeyValue = $null
$step5KeyFile = Join-Path $latestRun.FullName "05d-api-key-get-value.txt"
if (Test-Path $step5KeyFile) {
    $step5Text = Get-Content -Path $step5KeyFile -Raw
    if ($step5Text -match "(?ms)^OUTPUT:\s*(?<val>.+?)\s*$") {
        $candidate = $Matches["val"].Trim()
        if (-not [string]::IsNullOrWhiteSpace($candidate)) {
            $apiKeyValue = $candidate
            @(
                "COMMAND: reuse-step5-api-key-value",
                "EXIT_CODE: 0",
                "OUTPUT:",
                "Reused API key value extracted from 05d-api-key-get-value.txt"
            ) | Set-Content -Path (Join-Path $latestRun.FullName "07e0-proxy-api-key-value.txt")
        }
    }
}

if ([string]::IsNullOrWhiteSpace($apiKeyValue)) {
    $apiKeyValueRaw = Invoke-AwsStep -AwsArgs @("apigateway", "get-api-key", "--api-key", $apiKeyId, "--include-value", "--query", "value", "--output", "text") -OutputFile (Join-Path $latestRun.FullName "07e0-proxy-api-key-value.txt")
    $apiKeyValue = $apiKeyValueRaw.Trim()
}

$integrationInput = [ordered]@{
    restApiId = $restApiId
    resourceId = $proxyResourceId
    httpMethod = "GET"
    type = "HTTP"
    integrationHttpMethod = "GET"
    uri = $apiInvokeUrl
    requestParameters = @{
        "integration.request.header.x-api-key" = "'$apiKeyValue'"
    }
}
$integrationInputPath = Join-Path $latestRun.FullName "07e-proxy-put-integration-input.json"
$integrationInput | ConvertTo-Json -Depth 8 | Set-Content -Path $integrationInputPath

Invoke-AwsStep -AwsArgs @("apigateway", "put-integration", "--cli-input-json", "file://$integrationInputPath", "--output", "json") -OutputFile (Join-Path $latestRun.FullName "07e-proxy-put-integration.txt") | Out-Null
Invoke-AwsStep -AwsArgs @("apigateway", "put-method-response", "--rest-api-id", $restApiId, "--resource-id", $proxyResourceId, "--http-method", "GET", "--status-code", "200", "--output", "json") -OutputFile (Join-Path $latestRun.FullName "07f-proxy-put-method-response.txt") | Out-Null
Invoke-AwsStep -AwsArgs @("apigateway", "put-integration-response", "--rest-api-id", $restApiId, "--resource-id", $proxyResourceId, "--http-method", "GET", "--status-code", "200", "--output", "json") -OutputFile (Join-Path $latestRun.FullName "07g-proxy-put-integration-response.txt") | Out-Null
Invoke-AwsStep -AwsArgs @("apigateway", "create-deployment", "--rest-api-id", $restApiId, "--stage-name", "prod", "--output", "json") -OutputFile (Join-Path $latestRun.FullName "07h-proxy-deploy.txt") | Out-Null

$proxyPublicUrl = "https://$restApiId.execute-api.$effectiveRegion.amazonaws.com/prod/$ProxyPathPart"
$proxyPublicUrl | Set-Content -Path (Join-Path $latestRun.FullName "07i-proxy-public-url.txt")

$proxyStatus = "BLOCKED"
for ($attempt = 1; $attempt -le 18; $attempt++) {
    try {
        $proxyResponse = Invoke-RestMethod -Method Get -Uri $proxyPublicUrl -TimeoutSec 20
        $proxyResponse | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $latestRun.FullName "07j-proxy-public-invoke-response.json")
        $proxyStatus = "PASS"
        break
    }
    catch {
        if ($attempt -lt 18) {
            Start-Sleep -Seconds 10
            continue
        }
        @(
            "Proxy invoke failed or blocked."
            "Reason: $($_.Exception.Message)"
        ) | Set-Content -Path (Join-Path $latestRun.FullName "07j-proxy-public-invoke-response.json")
    }
}

$updatedSummary = [ordered]@{
    runId = $summary.runId
    accountId = $summary.accountId
    region = $summary.region
    bucket = $summary.bucket
    objectKey = $summary.objectKey
    lambdaFunctionName = $summary.lambdaFunctionName
    restApiId = $summary.restApiId
    apiInvokeUrl = $summary.apiInvokeUrl
    apiInvokeUrlRequiresApiKey = $summary.apiInvokeUrlRequiresApiKey
    apiKeyId = $summary.apiKeyId
    usagePlanId = $summary.usagePlanId
    apiInvokeAttempted = $summary.apiInvokeAttempted
    stepStatus = $summary.stepStatus
    evidenceDir = $summary.evidenceDir
    proxyPathPart = $ProxyPathPart
    proxyPublicUrl = $proxyPublicUrl
    proxyInvokeStatus = $proxyStatus
}
$updatedSummary | ConvertTo-Json -Depth 10 | Set-Content -Path $summaryPath

Write-Host "E2E + proxy setup finished."
Write-Host "Evidence directory: $($latestRun.FullName)"
Write-Host "Summary file: $summaryPath"
Write-Host "Proxy URL: $proxyPublicUrl"
