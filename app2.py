#---------------------Good Version - Use this

# --- Import Libraries ---
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Streamlit Setup ---
st.set_page_config(page_title="Warren's Fair Value Estimator", layout="wide")
st.title("📈 Warren's Fair Value Estimator")

# --- Input Section ---
col1, col2 = st.columns([2, 2])
with col1:
    st.markdown("<div style='padding-top: 200px;'></div>", unsafe_allow_html=True)
    ticker = st.text_input("Enter Ticker:", value="TCS").strip().upper()
    years = st.selectbox("Time Period:", options=[5, 10], index=0)
    discount = st.number_input("Discount Rate (%)", value=6.5, min_value=0.0, max_value=100.0) / 100
    terminal = st.number_input("Terminal Growth Rate (%)", value=2.0, min_value=0.0, max_value=100.0) / 100
    mos = st.number_input("Margin of Safety (%)", min_value=0, max_value=100, value=40) / 100

# --- Screener Data: Market Cap, Profit Growth ---
@st.cache_data(show_spinner=False)
def get_ratios(ticker):
    url = f'https://www.screener.in/company/{ticker}/consolidated/'
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        st.error(f"❌ Failed to retrieve data: {e}")
        return None, None, None, None

    # Market Cap & Current Price
    ratios = soup.find('ul', id='top-ratios')
    market_cap = current_price = None
    for li in ratios.find_all('li'):
        name = li.find('span', class_='name').text.strip()
        value = re.sub(r'[^\d.]', '', li.find('span', class_='value').text)
        if name == "Market Cap":
            market_cap = float(value)
        elif name == "Current Price":
            current_price = float(value)

    total_shares = market_cap / current_price if market_cap and current_price else None

    # Profit Growth
    profit_5yr = profit_10yr = None
    tables = soup.find_all("table", class_="ranges-table")
    for table in tables:
        if "Compounded Profit Growth" in table.text:
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if cells:
                    period = cells[0].text.strip()
                    val = cells[1].text.strip().replace('%', '')
                    rate = float(val) / 100 if val else 0
                    if "5" in period:
                        profit_5yr = rate
                    elif "10" in period:
                        profit_10yr = rate

    profit = profit_5yr if years == 5 else profit_10yr
    return total_shares, profit, current_price, market_cap

total_shares, profit, current_price, market_cap = get_ratios(ticker)

# --- Cashflow and CapEx ---
@st.cache_data(show_spinner=False)
def scrape_cashflow_and_capex(ticker):
    url = f"https://www.screener.in/company/{ticker}/consolidated/#cash-flow"
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("section", id="cash-flow").find("table", class_="data-table")
    headers = [th.text.strip() for th in table.find_all("th")]
    rows = [[td.text.strip() for td in tr.find_all("td")] for tr in table.find_all("tr")]
    df = pd.DataFrame(rows, columns=headers)

    # CFO
    cfo_row = df[df.iloc[:, 0].str.contains("Operating", case=False, na=False)]
    cfo_str_vals = cfo_row.iloc[0, 1:].tolist()  # ['1,50,000', '', '1,20,000', ...]
    cfo_vals = []
    for val in reversed(cfo_str_vals):
        if val.strip():  # Check if not empty or whitespace
            try:
                cfo_vals.append(float(val.replace(",", "")))
            except:
                cfo_vals.append(0.0)  # Handle any unexpected non-numeric value
        if len(cfo_vals) == 5:
            break

    # Reverse again to maintain chronological order (oldest to newest)
    cfo_vals.reverse()

    # CapEx (API)
    def get_company_id(ticker):
        page = requests.get(f"https://www.screener.in/company/{ticker}/consolidated/")
        soup = BeautifulSoup(page.content, "html.parser")
        return soup.find("div", {"id": "company-info"}).get("data-company-id")

    def get_capex(company_id):
        url = f"https://www.screener.in/api/company/{company_id}/schedules/?parent=Cash+from+Investing+Activity&section=cash-flow"
        res = requests.get(url)
        data = res.json().get('Fixed assets purchased', {})
        return [float(data[yr].replace(",", "")) for yr in sorted(data)[-5:]]

    capex = get_capex(get_company_id(ticker))
    while len(capex) < len(cfo_vals[-5:]):
        capex.insert(0, 0)

    fcf = [c - x for c, x in zip(cfo_vals[-5:], capex)]
    return fcf, cfo_vals[-5:], capex


fcf_list, cfo_vals, capex = scrape_cashflow_and_capex(ticker)

# --- CAGR ---
def adjusted_cagr(values):
    for i in range(len(values) - 1):
        if values[i] > 0:
            return ((values[-1] / values[i]) ** (1 / (len(values) - 1 - i))) - 1
    return 0

fcf_cagr = adjusted_cagr(fcf_list)
FCF = fcf_list[-1] if fcf_list else 0

# --- Max Growth Cap by Market Cap Category ---
def get_max_growth_cap(mcap):
    if mcap >= 900_000:
        return 0.15
    elif mcap >= 100_000:
        return 0.25
    elif mcap >= 30_000:
        return 0.40
    elif mcap >= 5_000:
        return 0.60
    elif mcap >= 100:
        return 1.00
    return 0.0

# --- Multi-Stage DCF Model ---
def multi_stage_dcf(fcf, init_growth, terminal_growth, wacc, years, max_cap):
    init_growth = min(init_growth, max_cap)
    growths = [init_growth - i * (init_growth - terminal_growth) / (years - 1) for i in range(years)]
    fcfs = []
    for i in range(years):
        fcf = fcf * (1 + growths[i])
        fcfs.append(fcf)
    dcf = sum([fcfs[i] / ((1 + wacc) ** (i + 1)) for i in range(years)])
    terminal = (fcfs[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
    dcf += terminal / ((1 + wacc) ** years)
    return dcf, fcfs, init_growth

# --- DCF with Profit Growth ---
if all([total_shares, profit, fcf_cagr, FCF, market_cap]):
    max_cap = get_max_growth_cap(market_cap)

    dcf_profit, fcfs_profit,init_growth = multi_stage_dcf(FCF, profit, terminal, discount, years, max_cap)
    print(f"Initial Growth Rate used: {init_growth*100:.2f}%")
    dcf_fcf, fcfs_fcf,init_growth = multi_stage_dcf(FCF, fcf_cagr, terminal, discount, years, max_cap)
    print(f"Initial Growth Rate used: {init_growth*100:.2f}%")

    price_profit = dcf_profit / total_shares
    price_fcf = dcf_fcf / total_shares
    low_fair, high_fair = sorted([price_fcf, price_profit])

    with col1:
        col10, col11 = st.columns(2)
        with col10:
            st.markdown(f"<h4 style='text-align:center; margin-bottom:0;'>🗠 Share Price</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center; margin-bottom:0;'>💰 Market Cap.</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center; margin-bottom:0;'>🫙 Category</h4>", unsafe_allow_html=True)

        with col11:
            st.markdown(f"<p style='font-size:24px; text-align:center; margin:7px 0;'>₹{current_price:.0f}</p>", unsafe_allow_html=True)
            if market_cap >= 900000:
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>₹{market_cap/100000:.1f}L Cr.</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>Mega-Cap</p>", unsafe_allow_html=True)
            elif market_cap >= 100000 and market_cap < 900000:
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>₹{market_cap/100000:.1f}L Cr.</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>Large-Cap</p>", unsafe_allow_html=True)
            elif market_cap >= 30000 and market_cap < 100000:
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>₹{market_cap/1000:.0f}K Cr.</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>Mid-Cap</p>", unsafe_allow_html=True)
            elif market_cap >= 5000 and market_cap < 30000:
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>₹{market_cap/1000:.0f}K Cr.</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>Small-Cap</p>", unsafe_allow_html=True)
            elif market_cap >= 100 and market_cap < 5000:
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>₹{market_cap:.0f} Cr.</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:24px; text-align:center; margin:8px 0;'>Micro-Cap</p>", unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Projected Free Cash Flows")

        fig, ax1 = plt.subplots()
        x = list(range(1, years + 1))

        # Plot FCF on primary y-axis (left)
        line1 = ax1.plot(x, fcfs_fcf, label="FCF CAGR", marker="o", color="blue")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Free Cash Flow", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")

        # Scale FCF y-axis automatically (or set manually if needed)
        ax1.set_ylim([min(fcfs_fcf)*0.9, max(fcfs_fcf)*1.1])

        # Create secondary y-axis (right) for profits
        ax2 = ax1.twinx()
        line2 = ax2.plot(x, fcfs_profit, label="Profit Growth Rate", marker="o", color="green")
        ax2.set_ylabel("Profit", color="green")
        ax2.tick_params(axis='y', labelcolor="green")

        # Scale profit y-axis automatically (or set manually if needed)
        ax2.set_ylim([min(fcfs_profit)*0.9, max(fcfs_profit)*1.1])


        # Determine if they are close to each other
        diff = abs(fcfs_profit[-1] - fcfs_fcf[-1])
        threshold = 0.05 * max(fcfs_profit[-1], fcfs_fcf[-1])  # 5% of the max value

        # Adjust offsets based on whether values are close
        if diff < threshold:
            profit_offset = 0.02 * fcfs_profit[-1]
            fcf_offset = -0.02 * fcfs_fcf[-1]
        else:
            profit_offset = 0
            fcf_offset = 0

        # Plot the text annotations with offsets
        ax2.text(
            years, fcfs_profit[-1] + profit_offset,
            f"{profit * 100:.1f}%",
            color="green", fontsize=8, fontweight="bold",
            ha="right", va="bottom"
        )

        ax1.text(
            years, fcfs_fcf[-1] + fcf_offset,
            f"{fcf_cagr * 100:.1f}%",
            color="blue", fontsize=8, fontweight="bold",
            ha="right", va="top"
        )

        # Combine legends from both axes
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        # Title and layout
        fig.suptitle(f"Projected FCF Over Time - {ticker.upper()}")
        fig.tight_layout()
        st.pyplot(fig)



    safe_profit = price_profit * (1 - mos)
    safe_fcf = price_fcf * (1 - mos)
    low_safe, high_safe = sorted([safe_fcf, safe_profit])

    # --- Summary ---
    st.markdown("<hr style='border: 1px solid #666; margin-top:40px;'>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"<h4 style='text-align:center; margin-bottom:0;'>🎯 Estimated Fair Price Range for {ticker}</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align:center; margin-top:5px;'>🛡️ Safe Price Range (with {int(mos*100)}% Margin of Safety)</h4>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<p style='font-size:24px; text-align:center; margin:5px 0;'>₹{low_fair:.0f}  -  ₹{high_fair:.0f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:24px; text-align:center; margin:5px 0; margin-top:18px;'>₹{low_safe:.0f}  -  ₹{high_safe:.0f}</p>", unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #666; margin-top:40px;'>", unsafe_allow_html=True)
    
    
    # ------------------------------------------------------------- Price Chart ------------------------------------------------------------------
    # st.subheader(f"📈 Interactive Price Chart for {ticker}")
    # # Time period selection
    
    

    # # EMA selection
    # col201, col202 = st.columns([2,2])
    # with col201:
    #     period = st.radio("Select Chart Period:", options=["1 Year", "3 Years", "5 Years"], horizontal=True)
    #     selected_period = {"1 Year": "1y", "3 Years": "3y", "5 Years": "5y"}[period]
    # with col202:
    #     ema_selection = st.multiselect("📊 Add EMAs to Chart", options=["EMA50", "EMA200", "EMA300"], default=[])
    # st.markdown("<hr style='border: 1.5px dashed rgba(102, 102, 102, 0.5); margin-top:10px;'>", unsafe_allow_html=True)

    # try:
    #     ticker_data = yf.Ticker(ticker + ".NS")
    #     hist = ticker_data.history(period=selected_period)

    #     if not hist.empty:
    #         # Compute EMAs
    #         if "EMA50" in ema_selection:
    #             hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
    #         if "EMA200" in ema_selection:
    #             hist["EMA200"] = hist["Close"].ewm(span=200, adjust=False).mean()
    #         if "EMA300" in ema_selection:
    #             hist["EMA300"] = hist["Close"].ewm(span=300, adjust=False).mean()

    #         fig = go.Figure()

    #         # Plot close price
    #         fig.add_trace(go.Scatter(
    #             x=hist.index,
    #             y=hist['Close'],
    #             mode='lines',
    #             name='Close Price',
    #             line=dict(color='#00BFFF')
    #         ))

    #         # Plot EMAs
    #         for ema in ema_selection:
    #             fig.add_trace(go.Scatter(
    #                 x=hist.index,
    #                 y=hist[ema],
    #                 mode='lines',
    #                 name=ema,
    #                 line=dict(width=1.5, dash='dot')
    #             ))

    #         # Highlight fair price ranges
    #         fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=low_fair, y1=high_fair, fillcolor="rgba(0, 255, 0, 0.2)", line_width=0)
    #         fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=low_safe, y1=high_safe, fillcolor="rgba(234, 239, 44, 0.3)", line_width=0)
    #         fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=low_safe, y1=low_safe*0.9, fillcolor="rgba(255, 165, 0 , 0.25)", line_width=0)

    #         # Layout
    #         fig.update_layout(
    #             title=f"{ticker} Stock Price ({period})",
    #             xaxis_title="Date",
    #             yaxis_title="Price (₹)",
    #             hovermode="x unified",
    #             dragmode="zoom",
    #             template="plotly_white",
    #             margin=dict(l=40, r=40, t=40, b=40)
    #         )

    #         st.plotly_chart(fig, use_container_width=True)
    #     else:
    #         st.warning("⚠️ No historical price data found for this ticker.")
    # except Exception as e:
    #     st.error(f"❌ Error loading interactive price chart: {e}")

    st.subheader(f"📈 Interactive Price Chart for {ticker}")

    # Time period selection and EMA selection in columns
    col201, col202 = st.columns([2, 2])
    with col201:
        period = st.radio("Select Chart Period:", options=["1 Year", "3 Years", "5 Years"], horizontal=True)
        selected_period = {"1 Year": "1y", "3 Years": "3y", "5 Years": "5y"}[period]
    with col202:
        ema_selection = st.multiselect("📊 Add EMAs to Chart", options=["EMA50", "EMA200", "EMA300"], default=[])

    # Dashed separator line
    st.markdown("<hr style='border: 1.5px dashed rgba(102, 102, 102, 0.5); margin-top:10px;'>", unsafe_allow_html=True)

    try:
        ticker_data = yf.Ticker(ticker + ".NS")
        hist = ticker_data.history(period=selected_period)

        if not hist.empty:
            # Compute EMAs
            if "EMA50" in ema_selection:
                hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
            if "EMA200" in ema_selection:
                hist["EMA200"] = hist["Close"].ewm(span=200, adjust=False).mean()
            if "EMA300" in ema_selection:
                hist["EMA300"] = hist["Close"].ewm(span=300, adjust=False).mean()

            fig = go.Figure()

            # Plot Close Price
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00BFFF')
            ))

            # Plot EMAs
            for ema in ema_selection:
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist[ema],
                    mode='lines',
                    name=ema,
                    line=dict(width=1.5, dash='dot')
                ))

            # Highlight Fair, Safe, Undervalued zones
            fig.add_shape(type="rect", xref="paper", yref="y",
                        x0=0, x1=1, y0=low_fair, y1=high_fair,
                        fillcolor="rgba(0, 255, 0, 0.2)", line_width=0, layer="below")

            fig.add_shape(type="rect", xref="paper", yref="y",
                        x0=0, x1=1, y0=low_safe, y1=high_safe,
                        fillcolor="rgba(234, 239, 44, 0.3)", line_width=0, layer="below")

            fig.add_shape(type="rect", xref="paper", yref="y",
                        x0=0, x1=1, y0=low_safe * 0.9, y1=low_safe,
                        fillcolor="rgba(165, 42, 42, 0.3)", line_width=0, layer="below")

            # Add invisible traces to force legend entries for value zones
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color="rgba(0, 255, 0, 0.8)", symbol="square"),
                name="Fair Value Zone"
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color="rgba(234, 239, 44, 0.7)", symbol="square"),
                name="Safe Value Zone"
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=15, color="rgba(165, 42, 42, 0.5)", symbol="square"),
                name="Undervalued Zone"
            ))

            # Layout customization
            fig.update_layout(
                title=f"{ticker} Stock Price ({period})",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode="x unified",
                dragmode="zoom",
                template="plotly_white",
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="top",      # anchor at top of legend box
                    y=-0.2,             # position below the x-axis (negative y moves down)
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(255,255,255,0)',  # transparent background
                    borderwidth=1
                )

            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No historical price data found for this ticker.")
    except Exception as e:
        st.error(f"❌ Error loading interactive price chart: {e}")


    
    st.markdown("<hr style='border: 1px solid #666; margin-top:10px;'>", unsafe_allow_html=True)
    # --- Debug: Print all fetched, scraped, calculated, projected variables ---
    with st.expander("🔍 Show All Variables (Debug Info)"):
        st.write("**Fetched/Scraped Variables:**")
        st.write({
            "ticker": ticker,
            "years": years,
            "discount": discount,
            "terminal": terminal,
            "mos": mos,
            "total_shares": total_shares,
            "profit": profit,
            "current_price": current_price,
            "market_cap": market_cap,
            "fcf_list": fcf_list,
        })
        st.write("**Calculated Variables:**")
        st.write({
            "fcf_cagr": fcf_cagr,
            "FCF (latest)": FCF,
            "max_cap": max_cap,
            "dcf_profit": dcf_profit,
            "dcf_fcf": dcf_fcf,
            "price_profit": price_profit,
            "price_fcf": price_fcf,
            "low_fair": low_fair,
            "high_fair": high_fair,
            "safe_profit": safe_profit,
            "safe_fcf": safe_fcf,
            "low_safe": low_safe,
            "high_safe": high_safe,
        })
        st.write("**Projected FCFs:**")
        st.write({
            "fcfs_profit": fcfs_profit,
            "fcfs_fcf": fcfs_fcf,
        })

            # --- Historical + Projected Table ---
    st.subheader("📄 Cash Flow Summary Table")

    with st.expander("🔽 View Cash Flow Summary (Historical & Projected)", expanded=False):

        # --- Historical: Get Year Labels ---
        latest_year = pd.Timestamp.now().year
        hist_years = [str(latest_year - i - 1) for i in reversed(range(len(fcf_list)))]
        proj_years = [f"Year {i}" for i in range(1, years + 1)]

        # --- Historical Table ---
        hist_data = {
            "Year": hist_years,
            "Operating Cash Flow (₹ Cr)": cfo_vals[-5:],
            "CapEx (₹ Cr)": capex,
            "Free Cash Flow (₹ Cr)": fcf_list
        }
        hist_df = pd.DataFrame(hist_data)

        # --- Projected Table ---
        proj_data = {
            "Year": proj_years,
            "Projected FCF (₹ Cr)": fcfs_fcf,
            "Projected Profit (₹ Cr)": fcfs_profit
        }
        proj_df = pd.DataFrame(proj_data)

        # --- Combined Table: Add Type Column ---
        hist_df["Type"] = "Historical"
        proj_df["Type"] = "Projected"

        # Rename columns to match for vertical stacking
        proj_df = proj_df.rename(columns={
            "Projected FCF (₹ Cr)": "Free Cash Flow (₹ Cr)",
            "Projected Profit (₹ Cr)": "Operating Cash Flow (₹ Cr)"
        })
        proj_df["CapEx (₹ Cr)"] = None  # Add blank CapEx column for projections

        # Combine
        combined_df = pd.concat([hist_df, proj_df], ignore_index=True)

        # Reorder columns
        combined_df = combined_df[["Year", "Type", "Operating Cash Flow (₹ Cr)", "CapEx (₹ Cr)", "Free Cash Flow (₹ Cr)"]]

        # --- Style Function for Highlighting Projected Rows ---
        def highlight_projected(row):
            if row["Type"] == "Projected":
                return ["background-color: rgba(225, 0, 0, 0.05)"] * len(row)  # light red
            return [""] * len(row)

        # --- Show Styled Table ---
        st.dataframe(
            combined_df.style
                .apply(highlight_projected, axis=1)
                .format({
                    "Operating Cash Flow (₹ Cr)": "{:.2f}",
                    "CapEx (₹ Cr)": "{:.2f}",
                    "Free Cash Flow (₹ Cr)": "{:.2f}"
                }),
            use_container_width=True
        )



else:
    st.warning("⚠️ Could not fetch all required data (e.g., total shares, profit). Please check the ticker symbol and try again.")

