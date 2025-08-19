# Test different detection parameters
$headers = @{
    "Content-Type" = "application/json"
}

$image_url = "https://img.remit.ee/api/file/BQACAgUAAyEGAASHRsPbAAKX2GikNkRtBNIpPGN6Vt984gSt1NfPAALAGgACDkAoVaHTlLruxLFJNgQ.jpg"

# Test different parameter combinations
$test_cases = @(
    @{ name = "Default"; conf = 0.25; iou = 0.45 },
    @{ name = "Strict Detection"; conf = 0.4; iou = 0.45 },
    @{ name = "Loose Detection"; conf = 0.15; iou = 0.45 },
    @{ name = "Reduce Duplicates"; conf = 0.25; iou = 0.3 },
    @{ name = "Allow Overlap"; conf = 0.25; iou = 0.6 }
)

foreach ($test in $test_cases) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Testing: $($test.name)" -ForegroundColor Yellow
    Write-Host "conf_threshold: $($test.conf), iou_threshold: $($test.iou)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    
    $body = @{
        image_url = $image_url
        model = "ResNet"
        conf_threshold = $test.conf
        iou_threshold = $test.iou
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8123/v1/predict" -Method POST -Headers $headers -Body $body
        
        if ($response.success) {
            Write-Host "Success" -ForegroundColor Green
            Write-Host "Objects detected: $($response.detection_result.detection_count)" -ForegroundColor White
            Write-Host "Protein: $($response.evaluation_result.protein)%" -ForegroundColor White
            Write-Host "Oil: $($response.evaluation_result.oil)%" -ForegroundColor White
            Write-Host "Total time: $($response.total_time_delta)s" -ForegroundColor White
        } else {
            Write-Host "Failed: $($response.error)" -ForegroundColor Red
        }
    } catch {
        Write-Host "Request failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 2  # Avoid too frequent requests
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Testing completed!" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
