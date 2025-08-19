# Test different detection parameters with detailed output
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

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Detection Parameters Comparison Test" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

foreach ($test in $test_cases) {
    Write-Host "`n----------------------------------------" -ForegroundColor Cyan
    Write-Host "Testing: $($test.name)" -ForegroundColor Yellow
    Write-Host "conf_threshold: $($test.conf), iou_threshold: $($test.iou)" -ForegroundColor Green
    Write-Host "----------------------------------------" -ForegroundColor Cyan
    
    $body = @{
        image_url = $image_url
        model = "ResNet"
        conf_threshold = $test.conf
        iou_threshold = $test.iou
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8123/v1/predict" -Method POST -Headers $headers -Body $body
        
        if ($response.success) {
            Write-Host "✅ SUCCESS" -ForegroundColor Green
            Write-Host "Objects detected: $($response.detection_result.detection_count)" -ForegroundColor White
            Write-Host "Protein: $($response.evaluation_result.protein)%" -ForegroundColor White
            Write-Host "Oil: $($response.evaluation_result.oil)%" -ForegroundColor White
            Write-Host "Total time: $($response.total_time_delta)s" -ForegroundColor White
            
            # Show detection details if available
            if ($response.detection_result.objects -and $response.detection_result.objects.Count -gt 0) {
                $avgConfidence = ($response.detection_result.objects | ForEach-Object { $_.confidence } | Measure-Object -Average).Average
                Write-Host "Average confidence: $([math]::Round($avgConfidence, 3))" -ForegroundColor Cyan
            }
        } else {
            Write-Host "❌ FAILED: $($response.error)" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ REQUEST FAILED: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 1  # Avoid too frequent requests
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Testing completed!" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nParameter Explanation:" -ForegroundColor Yellow
Write-Host "• conf_threshold: Minimum confidence for object detection (0.0-1.0)" -ForegroundColor White
Write-Host "  - Higher values = stricter detection, fewer false positives" -ForegroundColor Gray
Write-Host "  - Lower values = looser detection, more objects detected" -ForegroundColor Gray
Write-Host "• iou_threshold: IoU threshold for Non-Maximum Suppression (0.0-1.0)" -ForegroundColor White
Write-Host "  - Higher values = allow more overlapping boxes" -ForegroundColor Gray
Write-Host "  - Lower values = remove more overlapping boxes" -ForegroundColor Gray
