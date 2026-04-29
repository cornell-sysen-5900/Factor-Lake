# Supabase Credentials Setup Workflow

Use this guide to get the Streamlit app connected to the Supabase project on your local machine.

## 1. Open the project links

- [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm)
- [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake)
- [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/)

## 2. Confirm the credentials you will use

Your project should use the current Supabase values provided in the Supabase project:

- URL: `https://-----.supabase.co`
- Key: `sb_publishable_-----`

You can find them here:

- Click on the word `copy` underneath the project name in the main page and copy the URL.
- For the key, locate `Project settings` in the side bar -> then click `API Keys.`

## 3. Create your local `.env` file

1. Open the repo root.
2. Create a file named `.env` if it does not already exist.
3. Add these values to the file:

```env
SUPABASE_URL=your-url-here
SUPABASE_KEY=your-key-here
```

4. Save the file.
5. Confirm `.env` stays local and is not committed to git.

## 4. Start the app locally

1. Open a terminal in the repo root.
2. Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

3. Wait for the browser to open.
4. Confirm the app starts without credential errors.

## 5. Load data in the app

1. In the sidebar, select **Supabase (Cloud)** as the data source.
2. Click **Load Data**.
3. Wait for the success message.
4. Confirm the dataset loads into the app without errors.

## 6. Verify the setup worked

1. Restart the app once after editing `.env` if needed.
2. Re-open the app.
3. Click **Load Data** again.
4. Confirm the message says data loaded successfully.

## 7. Update credentials when they change

1. Edit `.env`.
2. Replace `SUPABASE_URL` and/or `SUPABASE_KEY` with the new values.
3. Save the file.
4. Restart `streamlit run app/streamlit_app.py`.
5. Load data again to confirm the new credentials work.

## 8. Use Colab only if you still need it

1. Open `colab_setup.ipynb`.
2. Confirm the notebook is still appropriate for your use case.
3. Run the notebook cells only if you are using the Colab path.

## 9. Keep credentials safe

1. Never commit real secrets.
2. Keep `.env` local.
3. Remove credentials from any shareable notebook or demo artifact.
4. Use the publishable key only in the allowed client-side context.

## 10. Troubleshoot the common failures

1. If you see a connection failure, check your internet access and the Supabase project status.
2. If you see an authentication failure, confirm the key is correct and not expired.
3. If the app cannot find `.env`, make sure it is in the repo root next to `app/`.

## 11. What success looks like

You are done when the app starts, **Supabase (Cloud)** loads data successfully, and the analysis workflow can continue without credential-related errors.

## 12. Reference

Use this section when you need to remember how the setup is wired.

1. Local development reads credentials from `.env` in the repo root.
2. `app/streamlit_app.py` loads the credentials automatically when the app starts.
3. The publishable key is the expected client-side key for the app workflow.
4. `colab_setup.ipynb` is only relevant if you are still using the Colab path.
5. The Streamlit app should only need the credentials you placed in `.env` or Streamlit Cloud secrets.

## 13. Reference links

- [Supabase Dashboard](https://supabase.com/dashboard)
- [Supabase Documentation](https://supabase.com/docs)
- [Python-dotenv Documentation](https://github.com/theskumar/python-dotenv)
