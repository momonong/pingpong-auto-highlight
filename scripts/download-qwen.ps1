# Alternative for Windows environments where Python Hub/Xet transport stalls.
# Downloads one fixed official snapshot, verifies LFS SHA-256, never uses local_dir.
$ErrorActionPreference = 'Stop'
$modelRoot = [IO.Path]::GetFullPath('D:\hf\_models')
$revision = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
$modelId = 'Qwen/Qwen3-VL-8B-Instruct'
$snapshot = Join-Path $modelRoot "hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/$revision"
$temporary = Join-Path $modelRoot 'tmp/qwen-download'
foreach ($target in @($snapshot, $temporary)) {
    if (-not ([IO.Path]::GetFullPath($target).StartsWith($modelRoot + '\', [StringComparison]::OrdinalIgnoreCase))) {
        throw 'Download target escaped model root'
    }
    $ancestor = [IO.DirectoryInfo]::new($target)
    while ($null -ne $ancestor) {
        if ($ancestor.Exists -and ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'Refusing a junction/symlink in the model cache directory chain'
        }
        $ancestor = $ancestor.Parent
    }
    New-Item -ItemType Directory -Force -Path $target | Out-Null
}
$metadata = Invoke-RestMethod "https://huggingface.co/api/models/$modelId/revision/${revision}?blobs=true"
if ($metadata.sha -ne $revision) { throw 'Model revision mismatch' }
$http = [Net.Http.HttpClient]::new()
$http.Timeout = [TimeSpan]::FromSeconds(30)
try {
    foreach ($file in $metadata.siblings) {
        $name = $file.rfilename
        if ($name -notmatch '\.(json|safetensors|txt|jinja)$' -and $name -notmatch '^(README.md|LICENSE.*)$') { continue }
        if ($name.Contains('/') -or $name.Contains('\')) { throw 'Unexpected nested model file' }
        $destination = Join-Path $snapshot $name
        if (Test-Path -LiteralPath $destination) {
            $existing = Get-Item -LiteralPath $destination
            if ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                if (-not $existing.ResolveLinkTarget($true).FullName.StartsWith($modelRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
                    throw 'Existing model link escaped cache root'
                }
            }
        }
        if ((Test-Path -LiteralPath $destination) -and (Get-Item -LiteralPath $destination).Length -eq $file.size) {
            if (-not $file.lfs -or (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower() -eq $file.lfs.sha256) {
                Write-Output "Reuse $name"
                continue
            }
            throw "Existing snapshot file failed checksum: $name"
        }
        if (Test-Path -LiteralPath $destination) { throw "Existing snapshot file has wrong size: $name" }
        $part = Join-Path $temporary "$name.part"
        Write-Output "Downloading $name ($($file.size) bytes)"
        $total = if (Test-Path -LiteralPath $part) { (Get-Item -LiteralPath $part).Length } else { 0L }
        if ($total -gt $file.size) { throw "Partial file too large: $name" }
        $last = $total
        $attempt = 0
        $outputStream = [IO.File]::Open($part, [IO.FileMode]::Append, [IO.FileAccess]::Write)
        try {
            while ($total -lt $file.size) {
                $stop = [Math]::Min($file.size - 1, $total + 8MB - 1)
                $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, "https://huggingface.co/$modelId/resolve/$revision/$name")
                $request.Headers.Range = [Net.Http.Headers.RangeHeaderValue]::new($total, $stop)
                try {
                    $response = $http.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
                    try {
                    $response.EnsureSuccessStatusCode() | Out-Null
                    if ([int]$response.StatusCode -ne 206 -or $response.Content.Headers.ContentRange.From -ne $total -or $response.Content.Headers.ContentRange.To -ne $stop) {
                        throw "Unexpected Range response for $name"
                    }
                    $inputStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
                    $deadline = [Threading.CancellationTokenSource]::new(60000)
                    try {
                        $buffer = [byte[]]::new(1MB)
                        while (($count = $inputStream.ReadAsync($buffer, 0, $buffer.Length, $deadline.Token).GetAwaiter().GetResult()) -gt 0) {
                            $outputStream.Write($buffer, 0, $count)
                            $total += $count
                            if ($total -gt $stop + 1) { throw "Range overflow for $name" }
                        }
                        if ($total -ne $stop + 1) { throw "Truncated Range for $name" }
                    } finally { $inputStream.Dispose(); $deadline.Dispose() }
                    } finally { $response.Dispose(); $request.Dispose() }
                } catch {
                    $attempt += 1
                    if ($attempt -gt 3) { throw "Range download failed after three retries at offset $total for $name" }
                    Write-Output "Retry $attempt for $name from byte $total after transport failure"
                    Start-Sleep -Seconds 2
                    continue
                }
                $attempt = 0
                if ($total - $last -ge 256MB) { Write-Output "$name $total / $($file.size)"; $last = $total }
            }
        } finally { $outputStream.Dispose() }
        $hash = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash.ToLower()
        if ($file.lfs -and $hash -ne $file.lfs.sha256) { throw "SHA-256 mismatch for $name" }
        # Both checked absolute paths are inside the same model root; this is a file rename.
        Move-Item -LiteralPath $part -Destination $destination
        Write-Output "Verified $name SHA256=$hash"
    }
    Write-Output "Complete snapshot: $snapshot"
} finally { $http.Dispose() }
