Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$zipFile = Join-Path (Get-Location) "triage-web-hardened.zip"
if (Test-Path $zipFile) { Remove-Item $zipFile -Force }

$mode = [System.IO.Compression.ZipArchiveMode]::Create
$archive = [System.IO.Compression.ZipFile]::Open($zipFile, $mode)
$basePath = (Get-Item "triage-web").FullName

Get-ChildItem -Path "triage-web" -Recurse -File | Where-Object { $_.FullName -notmatch "node_modules|\\\.next" } | ForEach-Object {
    $relPath = $_.FullName.Substring($basePath.Length + 1)
    $entryName = "triage-web/" + $relPath.Replace("\", "/")
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $_.FullName, $entryName) | Out-Null
}

$archive.Dispose()
$mb = ((Get-Item $zipFile).Length / 1MB).ToString("0.00")
Write-Host "Created archive: $zipFile ($mb MB)"
