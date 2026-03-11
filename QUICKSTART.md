# Quick Start Guide

This guide will help you get the project running in 5-10 minutes.

## Prerequisites

- Python 3.8+
- pip package manager
- Docker & Docker Compose (optional, for containerized deployment)
- Git (optional)

## Step 1: Install Dependencies (1-2 minutes)

```bash
# Navigate to project directory
cd ai-workflow-capstone-master

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Run Part 1 - Exploratory Data Analysis (5 minutes)

```bash
# Start Jupyter
jupyter notebook

# Open and run: notebooks/Part1_EDA.ipynb
# This explores the data and validates business hypotheses
```

**What you'll see:**
- Data loading from multiple JSON sources
- Revenue trends and patterns
- Correlation analysis
- Time-series features created

## Step 3: Run Part 2 - Model Development (3-5 minutes)

```bash
# Continue in Jupyter, open: notebooks/Part2_TimeSeries_Modeling.ipynb

# Or run directly:
# jupyter notebook notebooks/Part2_TimeSeries_Modeling.ipynb
```

**What happens:**
- Multiple models trained (Linear Regression, Random Forest, Gradient Boosting)
- Best model selected and hyperparameter tuned
- Model saved to `models/best_model.joblib`
- Performance metrics displayed

## Step 4: Start the API (2 minutes)

### Option A: Direct Python (easiest for testing)

```bash
# Open a new terminal/command prompt
python api/app.py

# API starts on http://localhost:5000
# Check health: curl http://localhost:5000/health
```

### Option B: Docker (recommended for production)

```bash
# Build and start container
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

## Step 5: Test the API (2 minutes)

```bash
# In a new terminal

# 1. Check health
curl http://localhost:5000/health

# 2. Get API status
curl http://localhost:5000/status

# 3. Train model (if not already trained)
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "./cs-train"}'

# 4. Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[1000, 50, 100, 5, 3000, 20, 150, 200, 300, 400, 500, 600, 700]]}'

# 5. Get metrics
curl http://localhost:5000/metrics
```

## Step 6: Run Unit Tests (1 minute)

```bash
# Test data ingestion, features, and API
python -m pytest tests/test_api.py -v

# Or with coverage
python -m pytest tests/test_api.py --cov=src
```

## Step 7: Run Part 3 - Deployment Analysis (3 minutes)

```bash
# In Jupyter: notebooks/Part3_API_Deployment.ipynb

# This notebook:
# - Tests API endpoints
# - Performs load testing
# - Analyzes production data
# - Generates final report
```

## Troubleshooting

### API won't start
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000  # Windows
lsof -i :5000                 # macOS/Linux

# Use different port
export FLASK_PORT=5001
python api/app.py
```

### Import errors
```bash
# Make sure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Docker issues
```bash
# Clean up containers
docker-compose down
docker system prune -a

# Rebuild
docker build -t revenue-prediction-api:1.0 .
docker-compose up -d
```

## Project Files Overview

```
📁 src/                      # Python modules
  ├── data_ingestion.py     # Load and prepare data
  ├── feature_engineering.py # Create ML features
  └── logger.py             # Training/prediction logs

📁 notebooks/                # Jupyter notebooks
  ├── Part1_EDA.ipynb       # Data exploration
  ├── Part2_TimeSeries_Modeling.ipynb  # Model building
  └── Part3_API_Deployment.ipynb       # API & deployment

📁 api/                      # Flask API
  └── app.py                # REST endpoints

📁 tests/                    # Unit tests
  └── test_api.py           # Comprehensive tests

📁 models/                   # Trained models
📁 logs/                     # Training/prediction logs
📁 data/                     # Processed data
📁 cs-train/                 # Training data (JSON)
📁 cs-production/            # Production data (JSON)

🐳 Dockerfile               # Docker image
🐳 docker-compose.yml       # Container orchestration
📄 requirements.txt         # Python dependencies
📄 PROJECT_README.md        # Full documentation
```

## What's Next?

1. **Explore the data** - Run Part 1 notebook to understand the business problem
2. **Build the model** - Run Part 2 to train and evaluate the forecasting model
3. **Deploy & monitor** - Run Part 3 to test the API and analyze predictions
4. **Customize** - Modify hyperparameters, features, or models as needed

## Key API Commands

```bash
# Training the model
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "./cs-train"}'

# Making predictions (single batch)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [1000, 50, 100, 5, 3000, 20, 150, 200, 300, 400, 500, 600, 700],
      [1100, 55, 110, 6, 3100, 21, 155, 210, 310, 410, 510, 610, 710]
    ]
  }'

# Get training history
curl http://localhost:5000/train_log | python -m json.tool

# Get prediction history
curl http://localhost:5000/predict_log | python -m json.tool

# Get latest metrics
curl http://localhost:5000/metrics | python -m json.tool
```

## Performance Notes

- **API Response**: ~50-150ms per request
- **Training Time**: ~2-5 minutes for full dataset
- **Model Size**: ~50MB (saved to disk)
- **Memory Usage**: ~500MB-1GB at runtime

## Documentation

For more detailed information:
- See `PROJECT_README.md` for complete documentation
- Check individual notebooks for detailed explanations
- Review source code comments for implementation details

---

**Status**: Ready to use! 🚀

For issues or questions, refer to the troubleshooting section above.
