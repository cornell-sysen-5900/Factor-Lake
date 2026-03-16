# 🚀 Deployment Guide - Streamlit Community Cloud

This guide walks you through deploying the Factor-Lake app to Streamlit Community Cloud (100% free!) so Cornell students can access it via a simple URL. (This is for SYSEN 5900 - Software Systems Engineering in Quant Finance students)

---

## 📋 Prerequisites

- ✅ GitHub account (free)
- ✅ Supabase account with Factor-Lake database
- ✅ Password you want to use for Cornell student access

---

## 🎯 Step 1: Prepare Your Secrets

1. **Copy the secrets template:**
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

2. **Edit `.streamlit/secrets.toml`** and fill in your actual values:
   ```toml
   password = "Cornell2024!"  # Choose a secure password
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-anon-public-key-here"
   ```

3. **Get your Supabase credentials:**
   - Go to [Supabase Dashboard](https://app.supabase.com)
   - Select your project
   - Settings → API
   - Copy "Project URL" and "anon public" key

4. **⚠️ NEVER commit secrets.toml to Git** - it's already in `.gitignore`

---

## 🌐 Step 2: Deploy to Streamlit Cloud

### A. Sign Up for Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click **"Sign up"** and connect your GitHub account
3. Authorize Streamlit to access your repositories
4. Secrets Configruation moves into the streamlit community UI


### B. Deploy Your App

1. Click **"New app"** button
2. Fill in the deployment form:
   - **Repository:** `https://github.com/cornell-sysen-5900/Factor-Lake.git`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** Choose a custom name (e.g., `cornell-factor-lake`)

3. Click **"Deploy!"**

### C. Configure Secrets

1. While the app is deploying, click **"⚙️ Settings"** in the top-right
2. Go to **"Secrets"** tab
3. Paste the contents of your `.streamlit/secrets.toml` file:
   ```toml
   password = "Cornell2024!"
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-anon-public-key-here"
   ```
4. Click **"Save"**
5. The app will automatically restart with the new secrets

---

## 🎓 Step 3: Share with Cornell Students

Your app is now live! Share these details:

**📧 Email Template:**
```
Subject: Access to Factor-Lake Portfolio Analysis Tool

Hi [Student Name],

The Factor-Lake Portfolio Analysis tool is now available online!

🔗 URL: https://cornell-factor-lake.streamlit.app
🔒 Password: test3

No installation required - just:
1. Click the link above
2. Enter the password
3. Start analyzing portfolios!

The app works on any device with a web browser (laptop, tablet, phone).

Questions? Reply to this email.

Best,
[Your Name]
```

---

## 🔒 Security Notes

### ✅ What's Protected:
- App access is password-gated
- Supabase credentials are encrypted in Streamlit Cloud
- Code remains in private/public GitHub repo (read-only for students)

### ⚠️ Important:
- Anyone with the password can access the app
- Don't share the password publicly (email only to Cornell students)
- Change password if it gets leaked (update in Streamlit Cloud secrets)

### 🔐 Enhanced Security (Optional):
If you need stronger access control later:
- Use [Streamlit Authenticator](https://github.com/mkhorasani/Streamlit-Authenticator) for user accounts
- Deploy to Google Cloud Run with [Identity-Aware Proxy](https://cloud.google.com/iap)
- Use Cornell's SSO/OAuth integration

---

## 🛠️ Step 4: Test Locally (Optional)

Before deploying, test the password protection locally:

1. **Create secrets file:**
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # Edit secrets.toml with your values
   ```

2. **Run Streamlit:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Test password gate:**
   - Open http://localhost:8501
   - You should see password prompt
   - Enter your password
   - App should load after successful auth

---

## 📊 Using Google Drive Excel Sheets - *(Currently Not Implemented as of Dec 1, 2025)*
*Still working on this aspect but eventually...*

~~Yes, Google Drive still works!~~

Students can:
1. Upload their Excel file to Google Drive
2. Get the shareable link
3. Paste it into the app (when using local Excel option)

**Note:** The "Local Excel" option is currently disabled in the UI (showing "🚧 Working on it"). To enable it:

1. Edit `app/streamlit_app.py`
2. Find the file uploader section (around line 261)
3. Change `disabled=True` to `disabled=False`
4. Commit and push to GitHub (Streamlit Cloud will auto-deploy)

---

## 🔄 Updating the App

Streamlit Cloud automatically deploys when you push to GitHub:

1. Make changes to your code locally
2. Test with `streamlit run app/streamlit_app.py`
3. Commit and push to `main` branch:
   ```bash
   git add .
   git commit -m "Update feature X"
   git push origin revamped_ux
   ```
4. Streamlit Cloud detects the push and redeploys automatically (takes ~1-2 minutes)

---

## 🐛 Troubleshooting

### App won't start?
- Check Streamlit Cloud logs (click "Manage app" → "Logs")
- Verify secrets are properly formatted (no extra quotes/spaces)
- Make sure `requirements.txt` includes all dependencies

### Password not working?
- Check secrets.toml syntax (no spaces around `=`)
- Re-save secrets in Streamlit Cloud dashboard
- Clear browser cache and try again

### Supabase connection failing?
- Verify SUPABASE_URL and SUPABASE_KEY are correct
- Check Supabase project is active (not paused)
- Test connection locally first

### Need help?
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Supabase Documentation](https://supabase.com/docs)

---

## 💰 Pricing (All Free!)

- ✅ **Streamlit Community Cloud:** Free forever
  - Unlimited public apps
  - 1 GB RAM, 1 CPU core per app
  - Perfect for Cornell class use

- ✅ **Supabase Free Tier:**
  - 500 MB database
  - 50,000 monthly active users
  - More than enough for Factor-Lake data

- ✅ **GitHub:** Free for public repositories

**Total Cost: $0/month** 🎉

---

## 📚 Additional Resources

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-community-cloud/get-started
- **Secrets Management:** https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- **Custom Domains:** https://docs.streamlit.io/streamlit-community-cloud/manage-your-app/custom-domains (optional)

---

**🎓 Ready to deploy? Follow steps 1-3 above and your Cornell students will have access in ~10 minutes!**
