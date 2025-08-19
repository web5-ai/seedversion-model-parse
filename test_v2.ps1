# 测试v2 API
$headers = @{
    "Content-Type" = "application/json"
}

$body = @{
    image_url = "https://img.remit.ee/api/file/BQACAgUAAyEGAASHRsPbAAKX2GikNkRtBNIpPGN6Vt984gSt1NfPAALAGgACDkAoVaHTlLruxLFJNgQ.jpg"
    model = "ResNet"
} | ConvertTo-Json

Write-Host "Testing v2 API..."
Write-Host "Request body: $body"

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8123/v2/predict" -Method POST -Headers $headers -Body $body
    Write-Host "Success: $response"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    Write-Host "Response: $($_.Exception.Response)"
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response Body: $responseBody"
    }
}
