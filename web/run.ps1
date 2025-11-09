# Run Streamlit Web App
# TMG Fall 2025

Write-Host "🚀 Starting Vietnamese Text Analysis Web App..." -ForegroundColor Green

# Check if in correct directory
if (-not (Test-Path "app.py")) {
    Write-Host "❌ Error: app.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the web/ directory" -ForegroundColor Yellow
    exit 1
}

# Check if streamlit is installed
$streamlit_check = pip list | Select-String "streamlit"
if (-not $streamlit_check) {
    Write-Host "⚠️ Streamlit not found. Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Run the app
Write-Host ""
Write-Host "✨ Launching Streamlit app..." -ForegroundColor Cyan
Write-Host "📱 App will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "⚡ Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

streamlit run app.py
