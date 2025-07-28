# Kaggle API Setup Guide

This guide will help you set up Kaggle API credentials to download real MNIST data for the ML Pipeline Cascade Monitor.

## Step 1: Create a Kaggle Account

1. Go to [Kaggle.com](https://www.kaggle.com)
2. Sign up for a free account
3. Verify your email address

## Step 2: Get Your API Credentials

1. **Log in to Kaggle**
2. **Go to your Account Settings**:
   - Click on your profile picture in the top right
   - Select "Account"
3. **Scroll down to "API" section**
4. **Click "Create New API Token"**
5. **Download the `kaggle.json` file**

The `kaggle.json` file contains your API credentials:
```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
```

## Step 3: Set Up Credentials

### Option A: Using kaggle.json file (Recommended)

1. **Create the Kaggle directory**:
   ```bash
   mkdir ~/.kaggle
   ```

2. **Move the kaggle.json file**:
   ```bash
   # On Windows:
   copy kaggle.json %USERPROFILE%\.kaggle\
   
   # On macOS/Linux:
   cp kaggle.json ~/.kaggle/
   ```

3. **Set proper permissions** (macOS/Linux only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Option B: Using Environment Variables

Set environment variables:
```bash
# On Windows:
set KAGGLE_USERNAME=your_kaggle_username
set KAGGLE_KEY=your_kaggle_api_key

# On macOS/Linux:
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
```

## Step 4: Test the Setup

1. **Install the Kaggle package** (if not already installed):
   ```bash
   pip install kaggle
   ```

2. **Test the connection**:
   ```bash
   kaggle datasets list
   ```

You should see a list of datasets if the setup is correct.

## Step 5: Run the Project with Real MNIST Data

1. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the tests**:
   ```bash
   python test_pipeline.py
   ```

3. **Launch the dashboard**:
   ```bash
   streamlit run app.py
   ```

4. **Select "Real MNIST Data"** in the dashboard sidebar

## Troubleshooting

### Common Issues:

1. **"Invalid API credentials"**:
   - Double-check your username and API key
   - Ensure the kaggle.json file is in the correct location
   - Verify your Kaggle account is active

2. **"Permission denied"** (macOS/Linux):
   - Run: `chmod 600 ~/.kaggle/kaggle.json`

3. **"Dataset not found"**:
   - The MNIST dataset is publicly available
   - Ensure you have internet connection
   - Try: `kaggle datasets download cdeotte/mnist-in-csv`

### Alternative: Manual Download

If the API doesn't work, you can manually download the MNIST data:

1. **Go to**: https://www.kaggle.com/datasets/cdeotte/mnist-in-csv
2. **Download**: `mnist_train.csv` and `mnist_test.csv`
3. **Place files** in the `data/` directory of the project

## Benefits of Using Real MNIST Data

- **Real-world relevance**: MNIST is a classic dataset used in production ML
- **Realistic drift patterns**: Image data shows realistic degradation over time
- **Visual interpretability**: You can see how "images" change with drift
- **Production experience**: Demonstrates handling of structured data
- **Resume impact**: Shows you can work with real datasets, not just synthetic data

## Data Description

The MNIST dataset contains:
- **70,000 handwritten digit images** (0-9)
- **28x28 pixel grayscale images**
- **60,000 training samples**, **10,000 test samples**
- **10 classes** (digits 0-9)

The system will:
1. **Download** the data automatically
2. **Preprocess** images (resize to 16x16 for faster processing)
3. **Simulate drift** (noise, blur, gradual changes)
4. **Monitor cascade effects** through the ML pipeline

## Security Note

- Keep your Kaggle API credentials secure
- Don't commit `kaggle.json` to version control
- The API key has limited permissions (read-only for public datasets) 