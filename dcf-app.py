import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(
    page_title="Warren's Fair Value Estimator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📈 Warren's Fair Value Estimator")

# Create columns for layout: inputs on left, chart on right
col1, col2 = st.columns([2, 2])

with col1:
    st.markdown("<div style='padding-top: 200px;'></div>", unsafe_allow_html=True)
    ticker = st.text_input("Enter Ticker:", value="TCS").strip().upper()
    years = st.selectbox("Time Period:", options=[5, 10], index=0)
    discount = st.number_input("Discount Rate (%)", value=6.5, min_value=0.0, max_value=100.0) / 100
    terminal = st.number_input("Terminal Growth Rate (%)", value=2.0, min_value=0.0, max_value=100.0) / 100
    mos = st.number_input("Margin of Safety (%)", min_value=0, max_value=100, value=40) / 100

@st.cache_data(show_spinner=False)
def get_ratios(ticker):
    url = f'https://www.screener.in/company/{ticker}/consolidated/'
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10) # Added timeout
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to retrieve data for {ticker}. Please try again later. Error: {e}")
        return None, None, None

    if response.status_code != 200:
        st.error("Failed to retrieve data.")
        return None, None, None

    soup = BeautifulSoup(response.content, 'html.parser')
    ratios_section = soup.find('ul', id='top-ratios')
    market_cap = current_price = None

    if not ratios_section:
        st.error("Could not find key ratios on the page. The page structure may have changed.")
        return None, None, None

    for item in ratios_section.find_all('li'):
        name = item.find('span', class_='name').get_text(strip=True)
        value = ''.join(item.find('span', class_='value').stripped_strings)
        value = re.sub(r'[^\d.]', '', value)

        if name == "Market Cap":
            try:
                market_cap = float(value)
            except:
                market_cap = None
        elif name == "Current Price":
            try:
                current_price = float(value)
            except:
                current_price = None

    total_shares = market_cap / current_price if market_cap and current_price else None

    profit_5yr = profit_10yr = None
    tables = soup.find_all("table", class_="ranges-table")
    for table in tables:
        header = table.find("th").get_text(strip=True)
        if "Compounded Profit Growth" in header:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if cells:
                    period = cells[0].get_text(strip=True)
                    val = cells[1].get_text(strip=True).replace('%', '') or '0'
                    try:
                        rate = float(val) / 100
                    except ValueError:
                        rate = 0
                    if period == "5 Years:":
                        profit_5yr = rate
                    elif period == "10 Years:":
                        profit_10yr = rate

    profit = profit_5yr if years == 5 else profit_10yr
    return total_shares, profit, current_price

total_shares, profit, current_price = get_ratios(ticker)

@st.cache_data(show_spinner=False)
def scrape_cashflow_and_capex(ticker):
    url = f"https://www.screener.in/company/{ticker}/consolidated/#cash-flow"
    try:
        response = requests.get(url, timeout=10) # Added timeout
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to retrieve cash flow data for {ticker}. Error: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    df = None
    section = soup.find("section", {"id": "cash-flow"})
    if section:
        table = section.find("table", {"class": "data-table"})
        if table:
            headers = [th.text.strip() for th in table.find("thead").find_all("th")]
            rows = table.find("tbody").find_all("tr")
            data = [[td.text.strip() for td in row.find_all("td")] for row in rows]
            df = pd.DataFrame(data, columns=headers)

    cfo_row = df[df.iloc[:, 0].str.contains("Cash from Operating", case=False, na=False)] if df is not None else pd.DataFrame()
    cfo_vals = []
    if not cfo_row.empty:
        cfo_vals = cfo_row.iloc[0, 1:].apply(lambda x: float(x.replace(",", "")) if x else 0).tolist()
    cfo_last5 = cfo_vals[-5:] if len(cfo_vals) >= 5 else cfo_vals

    def get_company_id(ticker):
        url = f"https://www.screener.in/company/{ticker}/consolidated/"
        try:
            res = requests.get(url, timeout=10) # Added timeout
        except requests.exceptions.RequestException:
            return None
        soup = BeautifulSoup(res.content, "html.parser")
        div = soup.find("div", {"id": "company-info"})
        return div.get("data-company-id") if div else None

    def get_capex(company_id):
        if company_id is None:
            return []
        url = f"https://www.screener.in/api/company/{company_id}/schedules/?parent=Cash+from+Investing+Activity&section=cash-flow"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            res = requests.get(url, headers=headers, timeout=10) # Added timeout
        except requests.exceptions.RequestException:
            return []
        if res.status_code != 200:
            return []
        data = res.json().get('Fixed assets purchased', {})
        sorted_years = sorted(data.keys())[-5:]
        return [float(data[y].replace(",", "")) for y in sorted_years]

    cid = get_company_id(ticker)
    capex_vals = get_capex(cid)

    while len(capex_vals) < len(cfo_last5):
        capex_vals.insert(0, 0)

    free_cash = [cfo - capex for cfo, capex in zip(cfo_last5, capex_vals)]

    return free_cash

fcf_list = scrape_cashflow_and_capex(ticker)

def adjusted_cagr(values):
    for i in range(len(values) - 1):
        if values[i] > 0:
            initial = values[i]
            final = values[-1]
            years_diff = len(values) - 1 - i
            return ((final / initial) ** (1 / years_diff) - 1) if years_diff > 0 else 0
    return 0

fcf_cagr = adjusted_cagr(fcf_list)
FCF = fcf_list[-1] if fcf_list else 0

def calculate_terminal_value(fcf, growth, long_growth, wacc, years):
    fcf_years = [fcf * ((1 + growth) ** (years - i)) for i in range(3)]
    avg_fcf = sum(fcf_years) / 3
    return (avg_fcf * (1 + long_growth)) / (wacc - long_growth)

if profit is not None and discount is not None and terminal is not None and years is not None and FCF is not None:
    tv1 = calculate_terminal_value(FCF, profit, terminal, discount, years)
    tv2 = calculate_terminal_value(FCF, fcf_cagr, terminal, discount, years)

    def discounted_cash_flow(fcf, growth, wacc, years, terminal_value):
        fcf_list = [fcf * ((1 + growth) ** t) for t in range(1, years + 1)]
        dcf = sum([cf / ((1 + wacc) ** t) for t, cf in enumerate(fcf_list, 1)])
        terminal = terminal_value / ((1 + wacc) ** years)
        return dcf + terminal, fcf_list

    dcf1, proj_fcf1 = discounted_cash_flow(FCF, profit, discount, years, tv1)
    dcf2, proj_fcf2 = discounted_cash_flow(FCF, fcf_cagr, discount, years, tv2)

    equity1 = dcf1
    equity2 = dcf2
    fair_price1 = equity1 / total_shares if total_shares else 0
    fair_price2 = equity2 / total_shares if total_shares else 0

    low_fair, high_fair = sorted([fair_price1, fair_price2])

    with col2:
        st.subheader("📊 Projected Free Cash Flows")
        fig, ax = plt.subplots()
        years_list = list(range(1, years + 1))
        ax.plot(years_list, proj_fcf1, label="Profit Growth Rate", marker="o", color="green")
        ax.plot(years_list, proj_fcf2, label="FCF CAGR", marker="o", color="blue")
        ax.text(years, proj_fcf1[-1], f"{profit * 100:.1f}% ", color="green", fontsize=8, fontweight="bold", ha="right", va="bottom")
        ax.text(years, proj_fcf2[-1], f"{fcf_cagr * 100:.1f}% ", color="blue", fontsize=8, fontweight="bold", ha="right", va="bottom")
        ax.set_xlabel("Year")
        ax.set_ylabel("Free Cash Flow")
        ax.set_title(f"Projected FCF Over Time - {ticker.upper()}")
        ax.legend()
        st.pyplot(fig)

    safe_price_low = low_fair * (1 - mos)
    safe_price_high = high_fair * (1 - mos)
    st.markdown("<hr style='border: 1px solid #666; margin-top:40px;'>", unsafe_allow_html=True)
    median_price1 = (low_fair + high_fair) / 2
    median_price2 = (safe_price_low + safe_price_high) / 2

    col3, col4, col5 = st.columns([3, 3, 3])

    with col3:
        st.markdown(f"<h4 style='text-align:center; margin-bottom:0;'>🎯 Estimated Fair Price Range for {ticker}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align:center; margin-top:5px;'>🛡️ Safe Price Range (with {int(mos*100)}% Margin of Safety)</h4>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<p style='font-size:24px; text-align:center; margin:5px 0;'>₹{low_fair:.2f} - ₹{high_fair:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:24px; text-align:center; margin:5px 0; margin-top:18px;'>₹{safe_price_low:.2f} - ₹{safe_price_high:.2f}</p>", unsafe_allow_html=True)

    with col5:
        st.markdown(f"<p style='font-size:24px; font-weight:bold; text-align:center; margin:5px 0;'>₹{median_price1:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:24px; font-weight:bold; text-align:center; margin:5px 0; margin-top:18px;'>₹{median_price2:.2f}</p>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #666; margin-top:40px;'>", unsafe_allow_html=True)
    st.subheader(f"📈 Interactive Price Chart for {ticker}")

    period = st.radio("Select Chart Period:", options=["1 Year", "3 Years", "5 Years"], horizontal=True)

    period_map = {
        "1 Year": "1y",
        "3 Years": "3y",
        "5 Years": "5y"
    }
    selected_period = period_map[period]

    try:
        ticker_data = yf.Ticker(ticker + ".NS")
        hist = ticker_data.history(period=selected_period)
        if not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00BFFF')
            ))
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=low_fair, y1=high_fair, fillcolor="rgba(0, 255, 0, 0.2)", line_width=0)
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=safe_price_low, y1=safe_price_high, fillcolor="rgba(234, 239, 44, 0.3)", line_width=0)
            fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=safe_price_low, y1=safe_price_low*0.75, fillcolor="rgba(255, 165, 0 , 0.25)", line_width=0)
            fig.update_layout(
                title=f"{ticker} Stock Price ({period})",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode="x unified",
                dragmode="zoom",
                template="plotly_white",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No historical price data found for this ticker.")
    except Exception as e:
        st.error(f"❌ Error loading interactive price chart: {e}")
else:
    st.warning("⚠️ Could not fetch all required data (e.g., total shares, profit). Please check the ticker symbol and try again.")