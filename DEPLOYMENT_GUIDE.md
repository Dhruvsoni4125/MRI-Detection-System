# Medical Disease Detection - Deployment Guide

## 🚀 Deployment Options

### 1. **Streamlit Cloud (Easiest - Recommended)**

#### Prerequisites:
- GitHub account
- Repository pushed to GitHub

#### Steps:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository: `your-username/Medical_Disease_Detection`
4. Set main file path: `app.py`
5. Click Deploy!

#### Advantages:
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ No server management
- ✅ Direct from GitHub

---

### 2. **Heroku Deployment**

#### Prerequisites:
- Heroku CLI installed
- Heroku account
- Git repository

#### Steps:
1. **Install Heroku CLI:**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku:**
   ```bash
   heroku login
   ```

3. **Create Heroku App:**
   ```bash
   heroku create your-medical-detection-app
   ```

4. **Set Buildpack for Python:**
   ```bash
   heroku buildpacks:set heroku/python
   ```

5. **Deploy:**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push heroku main
   ```

6. **Open App:**
   ```bash
   heroku open
   ```

#### Advantages:
- ✅ Free tier available
- ✅ Easy scaling
- ✅ Add-on ecosystem

---

### 3. **Docker Deployment**

#### Local Testing:
```bash
# Build Docker image
docker build -t medical-detection .

# Run locally
docker run -p 8501:8501 medical-detection
```

#### Deploy to Docker Hub:
```bash
# Tag and push to Docker Hub
docker tag medical-detection your-dockerhub-username/medical-detection
docker push your-dockerhub-username/medical-detection
```

#### Advantages:
- ✅ Portable across platforms
- ✅ Consistent environment
- ✅ Easy scaling with orchestration

---

### 4. **AWS EC2 Deployment**

#### Prerequisites:
- AWS account
- EC2 instance (t2.micro for free tier)

#### Steps:
1. **Launch EC2 Instance:**
   - AMI: Ubuntu 20.04 LTS
   - Instance Type: t2.micro (free tier)
   - Security Group: Allow ports 22 (SSH) and 8501 (Streamlit)

2. **Connect to Instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip nginx
   ```

4. **Clone Repository:**
   ```bash
   git clone https://github.com/your-username/Medical_Disease_Detection.git
   cd Medical_Disease_Detection
   ```

5. **Install Python Packages:**
   ```bash
   pip3 install -r requirements.txt
   ```

6. **Run Application:**
   ```bash
   nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
   ```

7. **Configure Nginx (Optional):**
   ```bash
   sudo nano /etc/nginx/sites-available/medical-detection
   ```

   Add:
   ```
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

#### Advantages:
- ✅ Full control
- ✅ Scalable
- ✅ Cost-effective

---

### 5. **Google Cloud Platform (GCP)**

#### Using App Engine:
1. **Install Google Cloud SDK**
2. **Create Project:**
   ```bash
   gcloud projects create medical-detection-app
   ```

3. **Create app.yaml:**
   ```yaml
   runtime: python39
   instance_class: F1
   automatic_scaling:
     target_cpu_utilization: 0.65
   handlers:
   - url: /.*
     script: auto
   ```

4. **Deploy:**
   ```bash
   gcloud app deploy
   ```

---

### 6. **Microsoft Azure**

#### Using Azure App Service:
1. **Install Azure CLI**
2. **Login:**
   ```bash
   az login
   ```

3. **Create Resource Group:**
   ```bash
   az group create --name medical-detection-rg --location eastus
   ```

4. **Create App Service:**
   ```bash
   az appservice plan create --name medical-detection-plan --resource-group medical-detection-rg --sku FREE
   az webapp create --name medical-detection-app --resource-group medical-detection-rg --plan medical-detection-plan --runtime "PYTHON:3.9"
   ```

5. **Deploy:**
   ```bash
   az webapp up --name medical-detection-app --resource-group medical-detection-rg --location eastus
   ```

---

## 📋 Pre-Deployment Checklist

- [ ] Test app locally: `streamlit run app.py`
- [ ] Ensure all dependencies are in `requirements.txt`
- [ ] Model files are present in `models/` directory
- [ ] Remove any sensitive data or API keys
- [ ] Update file paths to be relative
- [ ] Test with sample images
- [ ] Check model loading works correctly

## 🔒 Security Considerations

1. **Data Privacy:** Since this handles medical images, ensure HIPAA compliance if deploying for real use
2. **File Upload Security:** Implement file type validation and size limits
3. **Model Security:** Consider model encryption for sensitive medical AI models
4. **Access Control:** Add authentication if needed for medical applications

## 🚀 Recommended Deployment Path

**For Quick Deployment:** Streamlit Cloud (5 minutes)
**For Production:** Heroku or Docker + Cloud Platform
**For Enterprise:** Azure or AWS with proper security measures

## 📞 Support

If you encounter issues during deployment, check:
1. Application logs
2. Model file paths
3. Python version compatibility
4. Memory/CPU requirements