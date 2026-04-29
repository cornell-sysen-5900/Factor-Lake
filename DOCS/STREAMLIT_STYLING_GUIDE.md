# 🎨 Streamlit Customization Guide

Use this guide when you need to change the appearance of the [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/).

For platform-level styling guidance, refer to the [Streamlit docs](https://docs.streamlit.io/).

## Levels of Customization

### 1. **Theme Configuration** (.streamlit/config.toml)

```toml
[theme]
primaryColor = "#FF4B4B"        # Accent color (buttons, links)
backgroundColor = "#FFFFFF"      # Main background
secondaryBackgroundColor = "#F0F2F6"  # Sidebar, widgets
textColor = "#262730"           # Text color
font = "sans serif"             # Font family

# Additional color options you can use:
# - Blue theme: primaryColor = "#1f77b4"
# - Green theme: primaryColor = "#2ca02c"
# - Purple theme: primaryColor = "#9467bd"
# - Orange theme: primaryColor = "#ff7f0e"

# Dark mode:
base = "dark"  # or "light"
```

### 2. **Custom CSS** (In your Python code)

You can style almost everything with CSS:

```python
st.markdown("""
<style>
    /* Change button styles */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 15px 32px;
        font-size: 16px;
    }
    
    /* Change sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Change headers */
    h1 {
        color: #1f77b4;
        font-family: 'Georgia', serif;
    }
    
    /* Style dataframes */
    .dataframe {
        border: 2px solid #1f77b4 !important;
    }
    
    /* Change metric values */
    [data-testid="stMetricValue"] {
        color: #28a745;
        font-size: 3rem;
    }
    
    /* Animate elements */
    .stButton>button:hover {
        transform: scale(1.05);
        transition: 0.3s;
    }
    
    /* Add background image */
    .main {
        background-image: url('your-image-url.jpg');
        background-size: cover;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
```

### 3. **HTML Components** (Custom elements)

```python
# Custom cards
st.markdown("""
<div style="
    padding: 20px;
    border-radius: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
">
    <h2>Custom Card</h2>
    <p>Your content here</p>
</div>
""", unsafe_allow_html=True)

# Custom badges
st.markdown("""
<span style="
    background-color: #28a745;
    color: white;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
">NEW</span>
""", unsafe_allow_html=True)

# Progress bar with custom styling
st.markdown("""
<div style="
    width: 100%;
    background-color: #ddd;
    border-radius: 10px;
">
    <div style="
        width: 70%;
        background-color: #4CAF50;
        height: 30px;
        border-radius: 10px;
        text-align: center;
        line-height: 30px;
        color: white;
    ">70%</div>
</div>
""", unsafe_allow_html=True)
```

### 4. **Layout Customization**

```python
# Wide mode
st.set_page_config(layout="wide")

# Custom columns with different widths
col1, col2, col3 = st.columns([3, 1, 1])

# Containers
with st.container():
    st.write("Grouped content")

# Expanders
with st.expander("Click to expand"):
    st.write("Hidden content")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Tab 1", "📈 Tab 2", "💰 Tab 3"])
```

### 5. **Page Configuration**

```python
st.set_page_config(
    page_title="Your App Name",
    page_icon="🚀",  # emoji or image path
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded",  # or "collapsed"
    menu_items={
        'Get Help': 'https://your-help-url.com',
        'Report a bug': "https://your-bug-report-url.com",
        'About': "# Your App\nVersion 1.0"
    }
)
```

### 6. **Custom Fonts** (Google Fonts)

```python
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)
```

### 7. **Advanced: Custom Components**

You can create fully custom JavaScript components:

```bash
# Install component creator
pip install streamlit-component-lib

# Create custom component
streamlit-component-lib create my_component
```

### 8. **Third-Party Styling Libraries**

```python
# streamlit-extras
pip install streamlit-extras
from streamlit_extras.colored_header import colored_header
colored_header(
    label="My Header",
    description="Description",
    color_name="blue-70"
)

# streamlit-card
pip install streamlit-card
from streamlit_card import card

# streamlit-aggrid (beautiful tables)
pip install streamlit-aggrid
```

## 🎨 Pre-made Color Schemes

### Professional Blue
```python
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Dark Mode
```python
base = "dark"
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

### Mint Fresh
```python
primaryColor = "#00D4AA"
backgroundColor = "#F0FFF4"
secondaryBackgroundColor = "#C6F6D5"
textColor = "#1A202C"
```

### Sunset Orange
```python
primaryColor = "#FF6B35"
backgroundColor = "#FFF8F3"
secondaryBackgroundColor = "#FFE5D9"
textColor = "#2D3748"
```

### Purple Gradient
```python
primaryColor = "#9D4EDD"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F3E5F5"
textColor = "#240046"
```

## 📝 CSS Selector Cheat Sheet

```css
/* Main app container */
.main { }

/* Sidebar */
[data-testid="stSidebar"] { }

/* Buttons */
.stButton>button { }

/* Text input */
.stTextInput>div>div>input { }

/* Selectbox */
.stSelectbox { }

/* Checkbox */
.stCheckbox { }

/* Radio buttons */
.stRadio { }

/* Slider */
.stSlider { }

/* Dataframe */
.dataframe { }

/* Metric */
[data-testid="stMetricValue"] { }
[data-testid="stMetricDelta"] { }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { }
.stTabs [data-baseweb="tab"] { }

/* Headers */
h1, h2, h3 { }

/* Markdown */
.stMarkdown { }

/* Code blocks */
.stCodeBlock { }

/* Alerts */
.stAlert { }
.stSuccess { }
.stError { }
.stWarning { }
.stInfo { }
```

## 🚀 Quick Start: Apply to Your App

1. **Edit `.streamlit/config.toml`** - Change theme colors
2. **Edit CSS in `streamlit_app.py`** - Already enhanced!
3. **Run and reload** - Streamlit auto-updates on save

## 💡 Pro Tips

1. **Use browser DevTools** (F12) to inspect elements and find selectors
2. **Test responsiveness** with different screen sizes
3. **Keep it accessible** - maintain good contrast ratios
4. **Use CSS variables** for consistent theming
5. **Gradients** make things look modern:
   ```css
   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   ```

## 🔗 Resources

- [Streamlit Theme Editor](https://streamlit.io/generative_ai)
- [CSS Reference](https://www.w3schools.com/css/)
- [Color Palette Generator](https://coolors.co/)
- [Gradient Generator](https://cssgradient.io/)
