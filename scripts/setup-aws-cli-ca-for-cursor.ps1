# One-time fix: Avast HTTPS scanning breaks AWS CLI SSL inside Cursor terminals.
# Run this in an EXTERNAL terminal where `aws sts get-caller-identity` already works.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/setup-aws-cli-ca-for-cursor.ps1

$ErrorActionPreference = "Stop"

function Export-CertificateToPem {
    param(
        [System.Security.Cryptography.X509Certificates.X509Certificate2]$Certificate,
        [string]$Path
    )

    $derBytes = $Certificate.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Cert)
    $base64 = [Convert]::ToBase64String($derBytes)
    $lines = for ($i = 0; $i -lt $base64.Length; $i += 64) {
        $base64.Substring($i, [Math]::Min(64, $base64.Length - $i))
    }

    @("-----BEGIN CERTIFICATE-----") + $lines + @("-----END CERTIFICATE-----") |
        Set-Content -Path $Path -Encoding ascii
}

$certDir = Join-Path $env:USERPROFILE ".aws\certs"
New-Item -ItemType Directory -Force -Path $certDir | Out-Null

$awsCaSource = "C:\Program Files\Amazon\AWSCLIV2\awscli\botocore\cacert.pem"
if (-not (Test-Path $awsCaSource)) {
    throw "AWS CLI CA bundle not found at: $awsCaSource"
}

$avastCert = Get-ChildItem Cert:\LocalMachine\Root |
    Where-Object { $_.Subject -like "*Avast Web/Mail Shield Root*" } |
    Select-Object -First 1

if (-not $avastCert) {
    Write-Warning "Avast root certificate not found. If you use different AV software, add its root CA manually."
    exit 1
}

$avastPem = Join-Path $certDir "avast-root.pem"
$bundlePath = Join-Path $certDir "ca-bundle-with-avast.pem"

Export-CertificateToPem -Certificate $avastCert -Path $avastPem
Copy-Item $awsCaSource $bundlePath -Force
Add-Content -Path $bundlePath -Value "`n$(Get-Content $avastPem -Raw)" -Encoding ascii

aws configure set ca_bundle $bundlePath

Write-Host ""
Write-Host "Done. CA bundle written to:" -ForegroundColor Green
Write-Host "  $bundlePath"
Write-Host ""
Write-Host "Verify in Cursor terminal:" -ForegroundColor Green
Write-Host "  aws sts get-caller-identity"
