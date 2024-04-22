import pandas as pd
import streamlit as st
import time
import yfinance as yf
from datetime import datetime
import datetime
import math

secrets = st.session_state.secrets
page_config_set = False

def set_page_config():
    global page_config_set
    if not page_config_set:
        st.set_page_config(page_title="StockDash", page_icon="📈", layout="wide")
        page_config_set = True

set_page_config()
st.session_state.last_analysis_time = time.time() - 110

def highlight_gain_condition3(s):
    if s.name == 'Gain%':
        return s.apply(lambda x: highlight_gain_sell(x))

def highlight_gain_sell(x):
    if 3 < x <= 4:
        color = 'rgba(255, 140, 0, 1)'  # Orange with 50% opacity
    elif 4 < x:
        color = 'rgba(63, 255, 0,1)'  # Green with 50% opacity
    return 'background-color: %s' % color


def highlight_gain_condition(s):
    if s.name == 'ROI' or s.name == 'Gain':
        return s.apply(lambda x: highlight_single_gain(x))
    elif s.name == 'Total Investment':
        return s.apply(lambda x: highlight(x))
    else:
        return s.apply(lambda x: highlight_2(x))

def highlight_gain_condition2(s):
    if s.name == 'ROI':
        return s.apply(lambda x: highlight_roi(x))
    
def highlight_roi(value):
    if value < 0:
        color = 'rgba(255, 0, 0, 0.8)'  # Red with 50% opacity
    elif value == 0:
        color = 'rgba(255, 192, 203, 0.7)'
    elif 0 < value <= 2:
        color = 'rgba(255, 255, 0, 0.7)'  # Yellow with 50% opacity
    elif 2 < value <= 3:
        color = 'rgba(255, 140, 0, 1)'  # Orange with 50% opacity
    elif 3 < value:
        color = 'rgba(63, 255, 0,1)'  # Green with 50% opacity
    else:
        color = ''  # No highlighting if not in specified ranges
    return 'background-color: %s' % color

def highlight_gain(x):
    if 3 < x <= 4:
        color = 'rgba(255, 140, 0, 1)'  # Orange with 50% opacity
    elif 4 < x:
        color = 'rgba(63, 255, 0,1)'  # Green with 50% opacity
    return 'background-color: %s' % color

def highlight(x):
    color = 'rgba(139,190,27,1)'
    return 'background-color: %s' % color
def highlight_2(x):
    color = 'rgba(255, 140, 0, 1)'
    return 'background-color: %s' % color
def highlight_single_gain(value):
    if value <= 0:
        color = 'rgba(255, 0, 0, 0.8)'
    else:
        color = 'rgba(63, 255, 0,1)'  
    return 'background-color: %s' % color

def get_cmp_price(cmp_symbol):
    try:
        cmp_data = yf.Ticker(cmp_symbol+".NS")
        cmp_price = cmp_data.history(period="1d")["Close"].iloc[-1]
        return cmp_price
    except Exception as e:
        st.error(f"Failed to fetch cmp price: {e}")
        return None


if 'total_invested' not in st.session_state:
    st.session_state.total_invested = 0

sum_title = st.empty()
total_invested_place = st.empty()
sum_title.title('Summary')
col = st.columns(2)
col1 = col[0].empty()
col2 = col[1].empty()
headings = st.columns(2)
buy_head = headings[0].empty()
sell_head = headings[1].empty()
buy_sell = st.columns(2)
buy_Stock = buy_sell[0].empty()
sell_Stock = buy_sell[1].empty()
total_invested = 0
total_current_value = 0

def lifetime_high(ticker_symbol):
    stock_data = yf.Ticker(ticker_symbol + ".NS")
    historical_prices = stock_data.history(period="max")["High"]

    lifetime_high_price = historical_prices.max()

    return lifetime_high_price
while True:
    total_invested = 0
    total_current_value = 0
    investment_total = pd.DataFrame(columns=['Total Investment','Current Value','ROI','Gain'])
    investment_individual = pd.DataFrame(columns=["Stock",'Buy Avg', 'Qty','CMP', 'ROI','Gain','Total Investment','Current Value'])
    sell = pd.DataFrame(columns=['Stock', 'Price', 'Qty.', 'Age', 'CMP', 'Gain%', 'Amount'])
    buy = pd.DataFrame(columns=['Stock','Down%', 'Down_LB%',"LTH", 'Down_LTH%','CMP', 'LB','Amount', 'Qty'])
    if time.time() - st.session_state.last_analysis_time >= 0:
        st.session_state.last_analysis_time = time.time()
        # stocks = list(st.session_state.all_data.keys())
        stocks = secrets["connections"]["gsheets"]["worksheets"].values()
        today = datetime.datetime.today().date()
        for stock in stocks:
            time.sleep(1)
            up_df = st.session_state.all_data[stock]
            up_df['Stock'] = [stock] * up_df.shape[0]
            up_df['Price'] = up_df['Price'].str.replace(',', '').astype(float) if up_df['Price'].dtype == 'object' else up_df['Price']
            up_df['Qty.'] = up_df['Qty.'].str.replace(',', '').astype(float) if up_df['Qty.'].dtype == 'object' else up_df['Qty.']
            up_df['Age'] = (datetime.datetime.now() - pd.to_datetime(up_df['Date'])).dt.days
            up_df['CMP'] = round(get_cmp_price(st.session_state.secrets["connections"]["gsheets"]["worksheets"][stock]),2) if get_cmp_price(st.session_state.secrets["connections"]["gsheets"]["worksheets"][stock]) is not None else 1
            up_df['Gain%'] = round((((up_df['Qty.'] * up_df['CMP']) - (up_df['Price'] * up_df['Qty.'])) / (up_df['Price'] * up_df['Qty.'])) * 100,2)
            up_df['Amount'] = (up_df['Qty.'] * up_df['CMP']) - (up_df['Price'] * up_df['Qty.'])
            filtered_rows = up_df[up_df['Gain%'] >= 30]
            for Stock_name in filtered_rows['Stock'].unique():
                Stock_rows = filtered_rows[filtered_rows['Stock'] == Stock_name]
                Stock_rows.iloc[1:, 3] = ''  # Set Stock name to empty string for all rows except the first
                sell = pd.concat([sell, Stock_rows], ignore_index=True)
            st.session_state.all_data[stock]['Qty.'] = st.session_state.all_data[stock]['Qty.'].str.replace(',', '').astype(float) if st.session_state.all_data[stock]['Qty.'].dtype == 'object' else st.session_state.all_data[stock]['Qty.']
            cmp = round(get_cmp_price(st.session_state.secrets["connections"]["gsheets"]["worksheets"][stock]),2)
            total_value =  ((st.session_state.all_data[stock]['Qty.']) * (st.session_state.all_data[stock]['Price']).astype(float)).sum() if not st.session_state.all_data[stock].empty else 0
            total_invested += total_value
            current_value =  ((st.session_state.all_data[stock]['Qty.']) * cmp).sum() if not st.session_state.all_data[stock].empty else 0
            total_current_value += current_value
            total_qty = (st.session_state.all_data[stock]['Qty.']).sum() if not st.session_state.all_data[stock].empty else 1
            buy_price = round(total_value / total_qty,2)
            st.session_state.all_data[stock]['Price'] = pd.to_numeric(st.session_state.all_data[stock]['Price'], errors='coerce')
            last_buy = st.session_state.all_data[stock].sort_values('Date')['Price'].values[-1] if not st.session_state.all_data[stock].empty else 0
            pnl = (cmp-buy_price)/buy_price if buy_price != 0 else 0
            multi_fac = -1*round(pnl*1000,2)
            amount = 10000
            qty = math.ceil(amount / cmp)
            down_lb = round((cmp - last_buy)/last_buy * 100,2) if last_buy != 0 else 0
            lth = lifetime_high(st.session_state.secrets["connections"]["gsheets"]["worksheets"][stock])
            if last_buy == 0:
                if round((cmp-lth)/lth * 100,2) <= -15:
                    new_res = pd.DataFrame({'Stock': [stock], 'Down%':[round(pnl*100,2)], "Down_LTH%": [round((cmp - lth)/lth * 100,2)], "LTH": [lth], 'Down_LB%':[down_lb],'CMP':[cmp], 'Amount': [amount], 'Qty': [qty], 'LB': [last_buy]})
                    buy = pd.concat([buy,new_res],ignore_index=True)
            else:
                if down_lb <= -10:
                    new_res = pd.DataFrame({'Stock': [stock], 'Down%':[round(pnl*100,2)], "Down_LTH%": [round((cmp - lth)/lth * 100,2)], "LTH": [lth], 'Down_LB%':[down_lb],'CMP':[cmp], 'Amount': [amount], 'Qty': [qty], 'LB': [last_buy]})
                    buy = pd.concat([buy,new_res],ignore_index=True)
            if buy.empty:
                total = 0
            investment_individual = pd.concat([investment_individual,pd.DataFrame({"Stock":[stock], 'CMP':[cmp],'Buy Avg':[buy_price], 'Qty':[(st.session_state.all_data[stock]['Qty.']).sum()],'Total Investment':[total_value],'Current Value':[current_value],'ROI':[round((pnl) * 100,2)],'Gain':[round(current_value - total_value,2)]})],ignore_index=True)
        total = buy['Amount'].sum()
        format_dict2 = {'Price': '{:.2f}', 'Qty.': '{:.2f}', 'CMP': '{:.2f}', 'Gain%': '{:.2f}', 'Amount': '{:.2f}', 'Buy Value': '{:.2f}', 'Current Value': '{:.2f}'}
        if not sell.empty:
            sell.drop(columns=['Date'], axis = 1, inplace=True) 
        resultant_df_round = sell.round(2)
        styled_res_df = resultant_df_round.style.format(format_dict2).apply(highlight_gain_condition3, subset=['Gain%'], axis=0)
        if total_invested == 0:
            total_invested = 1
        investment_total = pd.concat([investment_total,pd.DataFrame({'Total Investment':[total_invested],'Current Value':[total_current_value],'ROI':[round(((total_current_value - total_invested)/total_invested) * 100,2)],'Gain':[round(total_current_value - total_invested,2)]})],ignore_index=True)
        res_rounded = investment_total.round(2)
        res_individual_rounded = investment_individual.sort_values("ROI", ascending=False).round(2)
        res_individual_rounded_1 = res_individual_rounded.iloc[:len(res_individual_rounded)//2]
        res_individual_rounded_2 = res_individual_rounded.iloc[len(res_individual_rounded)//2:]
        format_dict = {'Total Investment': '{:.2f}', 'CMP':'{:.2f}', 'Buy Avg':'{:.2f}','Qty': '{:.2f}', 'Current Value': '{:.2f}', 'ROI': '{:.2f}', 'Gain': '{:.0f}'}
        styled_res = res_rounded.style.format(format_dict).apply(highlight_gain_condition, axis=0)
        styled_res_individual_1 = res_individual_rounded_1.style.format(format_dict).apply(highlight_gain_condition2,subset=['ROI'], axis=0)
        styled_res_individual_2 = res_individual_rounded_2.style.format(format_dict).apply(highlight_gain_condition2,subset=['ROI'], axis=0)
        total_invested_place.dataframe(styled_res)
        numRows = len(res_individual_rounded)//2
        st.session_state.total_invested = total_invested
        col1.dataframe(styled_res_individual_1, use_container_width=True, height=(numRows + 1) * 35 + 3)
        col2.dataframe(styled_res_individual_2, use_container_width=True, height=(numRows + 2) * 35 + 3)
        buy_head.subheader('Buy')
        buy_Stock.dataframe(buy.sort_values('Down_LTH%'), use_container_width=True)
        sell_head.subheader('Sell')
        sell_Stock.dataframe(styled_res_df, use_container_width=True)
