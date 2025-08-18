# Restaurant Recommendation System - Test Suite

This comprehensive test suite ensures all components of the restaurant recommendation system work correctly with Google Places as the primary places API.

## 🏗️ Test Structure

```
tests/
├── conftest.py                    # Pytest configuration & fixtures
├── test_runner.py                 # Custom test runner with reporting
├── api_clients/
│   ├── test_google_places.py      # Google Places API integration tests
│   └── test_openai.py             # OpenAI client tests
├── databases/
│   ├── test_vector_db_mock.py     # Vector database tests
│   ├── test_cache_adapters.py     # Cache system tests
│   └── test_mock_data_quality.py  # Mock data quality validation
├── agents/
│   ├── test_query_parser.py       # Query parsing tests
│   ├── test_scoring.py            # Scoring algorithm tests
│   └── test_workflow.py           # End-to-end workflow tests
├── models/
│   └── test_data_models.py        # Pydantic model tests
└── integration/
    ├── test_api_comparison.py     # Mock vs real API comparison
    └── test_performance.py        # Performance benchmarks
```

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### 2. Set Environment Variables

```bash
# Optional - for real API testing
export GOOGLE_PLACES_API_KEY="your_google_places_key"
export OPENAI_API_KEY="your_openai_key"
```

### 3. Run Tests

```bash
# Run all tests with custom runner (recommended)
python tests/test_runner.py

# Run with pytest
pytest tests/

# Run specific test categories
python tests/api_clients/test_google_places.py
python tests/agents/test_workflow.py
python tests/databases/test_mock_data_quality.py
```

## 🧪 Test Categories

### **API Client Tests** (`api_clients/`)
- **Google Places API**: Real API integration, error handling, data quality
- **OpenAI Client**: LLM completion and embedding tests

**What they test:**
- API connectivity and authentication
- Response parsing and error handling
- Rate limiting and performance
- Data format consistency
- Mock vs real API behavior

### **Database Tests** (`databases/`)
- **Vector Database**: Similarity search, user personas, collaborative filtering
- **Cache System**: Memory/Redis adapters, TTL, performance
- **Mock Data Quality**: Ensures realistic test data

**What they test:**
- Data persistence and retrieval
- Vector similarity calculations
- Cache hit/miss ratios
- Mock data realism

### **Agent Tests** (`agents/`)
- **Query Parser**: Natural language → structured data
- **Scoring Algorithm**: 50/30/15/5 weight validation
- **Workflow**: End-to-end recommendation flow

**What they test:**
- Query understanding accuracy
- Recommendation scoring logic
- Workflow error handling
- Performance under load

### **Integration Tests** (`integration/`)
- **API Comparison**: Compare mock and real API results
- **Performance**: Load testing and benchmarks
- **User Scenarios**: Complete user journeys

## 🎯 Test Commands

### Run Individual Test Files
```bash
# Test Google Places API (requires API key)
python tests/api_clients/test_google_places.py

# Test complete workflow (works with mocks)
python tests/agents/test_workflow.py

# Test mock data quality
python tests/databases/test_mock_data_quality.py
```

### Run with Pytest Options
```bash
# Verbose output
pytest tests/ -v

# Run only fast tests (skip API calls)
pytest tests/ -m "not slow"

# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# Run in parallel
pytest tests/ -n 4
```

### Performance Testing
```bash
# Run performance benchmarks
pytest tests/integration/test_performance.py --benchmark-only

# Profile memory usage
pytest tests/ --profile
```

## 🔧 Test Configuration

### Environment-Specific Testing

```bash
# Test with mock services only (no API keys needed)
TESTING_MODE=mock python tests/test_runner.py

# Test with real APIs (requires API keys)
TESTING_MODE=real python tests/test_runner.py

# Mixed mode (real APIs where available, mocks elsewhere)
TESTING_MODE=mixed python tests/test_runner.py
```

### Custom Test Scenarios

The test runner supports custom scenarios:

```python
# Add to test_runner.py
async def test_custom_scenario(self):
    """Test specific business scenario"""
    workflow = await create_workflow(use_mock_services=True)
    
    result = await workflow.recommend_restaurants(
        user_query="Your specific test case",
        user_id="test_user",
        user_location=(40.7128, -74.0060)
    )
    
    # Your assertions here
    assert result["success"]
```

## 📊 Test Reports

### Custom Test Runner Output
```
🧪 Restaurant Recommendation System Test Suite
============================================================

📋 Testing API Clients
----------------------------------------
✅ Google Places API tests passed (Real API)
⚠️  OpenAI API key not found - skipping LLM tests

📋 Testing Database Systems
----------------------------------------
✅ Vector database tests passed
✅ Cache system tests passed

📊 TEST SUMMARY
============================================================
✅ Passed: 8
❌ Failed: 0
⚠️  Skipped: 1
📈 Total: 9
⏱️  Time: 3.45s
🎯 Success Rate: 88.9%
```

### Pytest HTML Report
```bash
pytest tests/ --html=reports/test_report.html
```

## 🐛 Debugging Test Failures

### Common Issues

1. **API Key Missing**
   ```
   GOOGLE_PLACES_API_KEY not available - using mock data
   ```
   **Fix**: Set environment variable or run in mock mode

2. **Network Timeouts**
   ```
   Request timeout after 30s
   ```
   **Fix**: Check internet connection or increase timeout

3. **Mock Data Inconsistency**
   ```
   AssertionError: Expected Italian restaurant, got Park
   ```
   **Fix**: Check mock data generation logic

### Debug Mode
```bash
# Run with debug logging
PYTHONPATH=src python tests/test_runner.py --debug

# Run single test with full output
pytest tests/api_clients/test_google_places.py::TestGooglePlacesClient::test_basic_search -v -s
```

## 🔄 Continuous Integration

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - run: pip install -r requirements-test.txt
    - run: python tests/test_runner.py
    env:
      GOOGLE_PLACES_API_KEY: ${{ secrets.GOOGLE_PLACES_API_KEY }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## 📈 Performance Benchmarks

Target performance metrics:
- **Query Processing**: < 2000ms average
- **API Response**: < 1000ms per call
- **Cache Hit Rate**: > 80%
- **Success Rate**: > 95%

## 🤝 Contributing Tests

When adding new features:

1. **Add unit tests** for new functions
2. **Add integration tests** for new workflows  
3. **Update mock data** if needed
4. **Verify performance** doesn't degrade

### Test Naming Convention
```python
def test_[component]_[feature]_[scenario]():
    """Test that [component] [does something] when [condition]"""
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Async Testing Guide](https://pytest-asyncio.readthedocs.io/)
- [Mock Data Best Practices](https://realpython.com/python-mock-library/)
- [Google Places API Documentation](https://developers.google.com/maps/documentation/places/web-service)

---

*Happy Testing! 🧪*