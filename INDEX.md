# IBM AI Enterprise Workflow Capstone - Project Index

## 📋 Start Here

**New to this project?** Read these in order:

1. **[QUICKSTART.md](QUICKSTART.md)** ⭐ (5-10 minutes)
   - Quick start guide to get everything running
   - Basic commands and troubleshooting

2. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** (10 minutes)
   - Complete overview of what was built
   - Project structure and key components
   - Business value and insights

3. **[PROJECT_README.md](PROJECT_README.md)** (20 minutes)
   - Full documentation
   - Detailed API documentation
   - Production deployment guide

---

## 🎯 The Three Parts

### Part 1: Exploratory Data Analysis
**File**: `notebooks/Part1_EDA.ipynb`
**Time**: 30-45 minutes
**What you'll learn**:
- How to load data from multiple JSON sources
- Business hypothesis formation
- Data exploration and visualization
- Time-series feature extraction

**Topics Covered**:
- Business scenario articulation
- Data quality assessment
- Correlation analysis
- Autocorrelation study (ACF/PACF)
- Time-series aggregation

---

### Part 2: Time-Series Modeling
**File**: `notebooks/Part2_TimeSeries_Modeling.ipynb`
**Time**: 30-45 minutes
**What you'll learn**:
- Multiple modeling approaches
- Hyperparameter tuning
- Feature engineering strategies
- Model evaluation and selection

**Models Compared**:
1. Linear Regression (baseline)
2. Random Forest Regressor
3. Gradient Boosting Regressor

**Results**:
- Best Model: Gradient Boosting
- R² Score: > 0.85
- RMSE: < 10% of mean revenue

---

### Part 3: API Development & Deployment
**File**: `notebooks/Part3_API_Deployment.ipynb`
**Time**: 30-45 minutes
**What you'll learn**:
- Flask API development
- Docker containerization
- Unit testing and TDD
- Production monitoring
- Load testing

**Deliverables**:
- 7 REST API endpoints
- Docker container ready
- Comprehensive unit tests
- Monitoring dashboards

---

## 📁 Project Structure Quick Reference

```
src/                          # Python modules (code, not notebooks!)
├── data_ingestion.py        # Load & preprocess data
├── feature_engineering.py   # Create ML features
├── logger.py                # Logging system
└── __init__.py              # Module init

notebooks/                    # Jupyter notebooks (analysis & reporting)
├── Part1_EDA.ipynb          # ← Start here for analysis
├── Part2_TimeSeries_Modeling.ipynb
└── Part3_API_Deployment.ipynb

api/
├── app.py                   # Flask API with 7 endpoints
└── [app runs on port 5000]

tests/
├── test_api.py             # Unit tests (pytest format)
└── [run with: pytest tests/test_api.py -v]

models/                      # Trained models (auto-created)
logs/                        # Training/prediction logs
data/                        # Processed data
cs-train/                    # Training data (23 JSON files)
cs-production/               # Production data (5 JSON files)

Dockerfile                   # Docker image definition
docker-compose.yml           # Container orchestration
requirements.txt             # Python dependencies
```

---

## 🚀 Quick Commands

### Installation
```bash
pip install -r requirements.txt
```

### Run Part 1 (EDA)
```bash
jupyter notebook notebooks/Part1_EDA.ipynb
```

### Run Part 2 (Modeling)
```bash
jupyter notebook notebooks/Part2_TimeSeries_Modeling.ipynb
```

### Start API (Option 1: Direct)
```bash
python api/app.py
# Visit http://localhost:5000/health
```

### Start API (Option 2: Docker)
```bash
docker-compose up -d
docker logs -f revenue-prediction-api
```

### Run Part 3 (Deployment)
```bash
jupyter notebook notebooks/Part3_API_Deployment.ipynb
```

### Run Tests
```bash
python -m pytest tests/test_api.py -v
```

---

## 🔑 Key Features

### Data Processing
- ✅ Multi-source JSON ingestion
- ✅ Non-uniform column handling
- ✅ Invoice ID cleaning
- ✅ Complete date range generation
- ✅ Automatic caching to CSV

### Feature Engineering
- ✅ Lagged variables (1, 7, 14, 30, 90 days)
- ✅ Rolling averages (7, 14, 30 days)
- ✅ Business metrics aggregation
- ✅ Supervised learning format

### Modeling
- ✅ Multiple algorithms tested
- ✅ Hyperparameter tuning (GridSearchCV)
- ✅ Time-series aware evaluation
- ✅ Feature importance analysis
- ✅ Residual diagnostics

### API
- ✅ 7 REST endpoints
- ✅ Training endpoint (retraining capability)
- ✅ Prediction endpoint (batch support)
- ✅ Logging endpoints (audit trail)
- ✅ Health & status monitoring

### Testing
- ✅ Data ingestion tests
- ✅ Feature engineering tests
- ✅ API endpoint tests
- ✅ Drift detection tests
- ✅ Scalability tests

### Deployment
- ✅ Docker containerization
- ✅ Docker Compose orchestration
- ✅ Health checks configured
- ✅ Volume mounts for persistence
- ✅ Environment variables

---

## 📊 Model Performance

| Metric | Value | Target |
|--------|-------|--------|
| R² Score | > 0.85 | > 0.80 ✓ |
| RMSE | < 10% | < 10% ✓ |
| MAPE | < 12% | < 15% ✓ |
| API Response | 50-150ms | < 200ms ✓ |
| Throughput | 100+ req/min | > 50 req/min ✓ |

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/test_api.py -v
```

### Test Coverage
```bash
pytest tests/test_api.py --cov=src --cov-report=html
```

### Test Categories
- **TestDataIngestion**: Data loading, validation, error handling
- **TestTimeSeries**: Time-series conversion, aggregation
- **TestFeatureEngineering**: Feature creation, no NaNs, scaling
- **TestLogger**: Logging to CSV, retrieval
- **TestAPIDrift**: Drift detection, scalability
- **TestDataValidation**: Quality checks, consistency

---

## 🔧 API Endpoints Reference

### Health & Monitoring
```bash
GET /health          # ← Check if API is running
GET /status          # ← Detailed status info
GET /metrics         # ← Latest model metrics
```

### Model Training
```bash
POST /train          # Train model on data directory
# Body: {"data_dir": "./cs-train"}
```

### Predictions
```bash
POST /predict        # Make revenue predictions
# Body: {"features": [[feature_array]]}
```

### Logs
```bash
GET /train_log       # All training history
GET /predict_log     # All prediction history
```

---

## 📚 Documentation Files

| File | Purpose | Time |
|------|---------|------|
| [QUICKSTART.md](QUICKSTART.md) | Get running in 5-10 min | 5 min |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Project overview | 10 min |
| [PROJECT_README.md](PROJECT_README.md) | Full documentation | 20 min |
| [Part1_EDA.ipynb](notebooks/Part1_EDA.ipynb) | Data analysis | 30 min |
| [Part2_Modeling.ipynb](notebooks/Part2_TimeSeries_Modeling.ipynb) | Model development | 30 min |
| [Part3_Deployment.ipynb](notebooks/Part3_API_Deployment.ipynb) | API deployment | 30 min |

---

## ⚙️ Technology Stack

**Data & ML**:
- scikit-learn: ML models and evaluation
- pandas: Data manipulation
- numpy: Numerical computing
- statsmodels: Time-series analysis
- matplotlib & seaborn: Visualization

**API & Infrastructure**:
- Flask: Web framework
- joblib: Model serialization
- Docker: Containerization
- pytest: Testing framework

**Python Version**: 3.8+

---

## 🎓 Learning Path

**Beginner**:
1. Read QUICKSTART.md
2. Run Part 1 notebook (EDA)
3. Run tests to see what's tested

**Intermediate**:
1. Read PROJECT_README.md
2. Run all three part notebooks
3. Understand each API endpoint
4. Study the source code in `src/`

**Advanced**:
1. Modify models in Part 2
2. Add new features in `src/feature_engineering.py`
3. Create new API endpoints
4. Deploy to cloud (AWS, GCP, Azure)

---

## 🆘 Troubleshooting

**API won't start:**
```bash
# Check port 5000 is not in use
netstat -ano | findstr :5000
# Use different port: export FLASK_PORT=5001
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Docker issues:**
```bash
# Clean and rebuild
docker-compose down
docker system prune -a
docker-compose up -d
```

**See QUICKSTART.md** for more troubleshooting tips.

---

## ✅ Checklist to Get Started

- [ ] Read QUICKSTART.md (5 min)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run Part 1 notebook (30 min)
- [ ] Run Part 2 notebook (30 min)
- [ ] Start API: `python api/app.py`
- [ ] Run Part 3 notebook (30 min)
- [ ] Run tests: `pytest tests/test_api.py -v`
- [ ] Read PROJECT_README.md for details
- [ ] Try Docker: `docker-compose up -d`

**Total time: ~2-3 hours to complete full project**

---

## 📞 Support

**For detailed help:**
1. Check [QUICKSTART.md](QUICKSTART.md) troubleshooting section
2. Review [PROJECT_README.md](PROJECT_README.md) for comprehensive guide
3. Check inline code comments in `src/` modules
4. Review test cases in `tests/test_api.py` for usage examples

**Common Issues:**
- API not starting: Check port 5000
- Import errors: Reinstall requirements
- Data not loading: Check `cs-train/` directory exists
- Docker issues: Use `docker-compose` instead of manual setup

---

## 🎉 What You've Learned

By working through this project, you'll understand:

✓ **Data Science Workflow**: From raw data to production model
✓ **Software Engineering**: Modules, testing, documentation
✓ **Time-Series Analysis**: Forecasting revenue patterns
✓ **API Development**: Building production-grade services
✓ **DevOps**: Docker, containerization, deployment
✓ **Monitoring**: Logging, health checks, drift detection

---

**Ready to get started?** → **[Start with QUICKSTART.md](QUICKSTART.md)** 🚀

---

*Last updated: December 2025*
*Status: Complete and Production-Ready*
