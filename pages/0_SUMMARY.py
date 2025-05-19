import math
from typing import Dict, Tuple
from datetime import datetime
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Constants
REFRESH_INTERVAL = 5  # seconds
GAIN_THRESHOLD = 5  # percentage for sell condition
DOWNSIDE_THRESHOLD = -5  # percentage for buy condition
LTH_DOWNSIDE_THRESHOLD = -90  # percentage from lifetime high for buy condition
DURATION = 30 # last buy se > 30 days


class ETFDashboard:
    def __init__(self):
        self.secrets = st.session_state.secrets
        self._initialize_session_state()
        self._setup_ui_containers()

    def _initialize_session_state(self) -> None:
        if 'total_invested' not in st.session_state:
            st.session_state.total_invested = 0
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = time.time() - (REFRESH_INTERVAL + 10)

    def run(self) -> None:
        while True:
            if time.time() - st.session_state.last_analysis_time >= REFRESH_INTERVAL:
                st.session_state.last_analysis_time = time.time()
                self._update_dashboard()

    def _setup_ui_containers(self) -> None:
        self.sum_title = st.empty()
        self.total_invested_place = st.empty()

        self.sum_title.title('Summary')
        cols = st.columns(2)
        self.col1 = cols[0].empty()
        self.col2 = cols[1].empty()

        headings = st.columns(2)
        self.buy_head = headings[0].empty()
        self.sell_head = headings[1].empty()

        buy_sell = st.columns(2)
        self.buy_etf = buy_sell[0].empty()
        self.sell_etf = buy_sell[1].empty()

    def _update_dashboard(self) -> None:
        investment_total, investment_individual, buy_df, sell_df = self._analyze_portfolio()
        self._display_summary(investment_total)
        self._display_individual_investments(investment_individual)
        self._display_buy_sell_recommendations(buy_df, sell_df)

    def _analyze_portfolio(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        total_invested = 0
        total_current_value = 0

        investment_total = pd.DataFrame(
            columns=['Total Investment', 'Current Value', 'ROI', 'Gain'])
        investment_individual = pd.DataFrame(columns=[
            "ETF", 'Buy Avg', 'Qty', 'CMP', 'ROI', 'Gain',
            'Total Investment', 'Current Value'
        ])
        sell_df = pd.DataFrame(
            columns=['ETF', 'Price', 'Qty.', 'Age', 'CMP', 'Gain%', 'Amount'])
        buy_df = pd.DataFrame(columns=[
            'ETF', 'Down%', 'Down_LB%', "LTH", 'Down_LTH%',
            'CMP', 'LB', 'Amount', 'Qty'
        ])

        for stock, stock_data in st.session_state.all_data.items():
            if stock_data.empty:
                stock_data = pd.DataFrame([{
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Price": "0",
                    "Qty.": "1"
                }])
            time.sleep(5)
            processed_data = self._process_stock_data(stock, stock_data)
            sell_df = self._update_sell_recommendations(
                sell_df, processed_data['sell_candidates'])

            total_invested += processed_data['total_value']
            total_current_value += processed_data['current_value']

            buy_df = self._update_buy_recommendations(
                buy_df, stock, processed_data)
            investment_individual = pd.concat([
                investment_individual,
                processed_data['individual_investment']
            ], ignore_index=True)

        investment_total = self._update_total_investment(
            investment_total, total_invested, total_current_value)
        st.session_state.total_invested = total_invested
        return investment_total, investment_individual, buy_df, sell_df

    def _process_stock_data(self, stock: str, stock_data: pd.DataFrame) -> Dict:
        processed_data = stock_data.copy()
        processed_data['ETF'] = stock
        numeric_cols = ['Price', 'Qty.']
        for col in numeric_cols:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].str.replace(
                    ',', '').astype(float)

        processed_data['Age'] = (
            datetime.now() - pd.to_datetime(processed_data['Date'])).dt.days
        ticker = self.secrets["connections"]["gsheets"]["worksheets"][stock]
        processed_data['CMP'] = round(self._get_cmp_price(ticker), 2)
        processed_data['Gain%'] = round(
            ((processed_data['Qty.'] * processed_data['CMP'] -
              processed_data['Price'] * processed_data['Qty.']) /
             (processed_data['Price'] * processed_data['Qty.']) * 100), 2
        )
        processed_data['Amount'] = (
            processed_data['Qty.'] * processed_data['CMP'] -
            processed_data['Price'] * processed_data['Qty.']
        )
        sell_candidates = processed_data[
    (processed_data['Gain%'] > GAIN_THRESHOLD) & (processed_data['Price'] > 0)
]


        total_value = processed_data['Qty.'].dot(processed_data['Price'])
        total_qty = processed_data['Qty.'].sum()
        buy_price = round(total_value / total_qty, 2) if total_qty != 0 else 0
        current_value = processed_data['Qty.'].dot(processed_data['CMP'])
        pnl = (processed_data['CMP'].iloc[0] - buy_price) / \
            buy_price if buy_price != 0 else 0

        individual_investment = pd.DataFrame({
            "ETF": [stock],
            'CMP': [processed_data['CMP'].iloc[0]],
            'Buy Avg': [buy_price],
            'Qty': [total_qty],
            'Total Investment': [total_value],
            'Current Value': [current_value],
            'ROI': [round(pnl * 100, 2)],
            'Gain': [round(current_value - total_value, 2)]
        })

        return {
            'age': processed_data['Age'],
            'processed_data': processed_data,
            'sell_candidates': sell_candidates,
            'total_value': total_value,
            'current_value': current_value,
            'buy_price': buy_price,
            'pnl': pnl,
            'individual_investment': individual_investment,
            'last_buy_price': processed_data['Price'].iloc[-1] if not processed_data.empty else 0,
            'cmp': processed_data['CMP'].iloc[0] if not processed_data.empty else 0
        }

    def _update_sell_recommendations(self, sell_df: pd.DataFrame, sell_candidates: pd.DataFrame) -> pd.DataFrame:
        if not sell_candidates.empty:
            for etf_name in sell_candidates['ETF'].unique():
                etf_rows = sell_candidates[sell_candidates['ETF'] == etf_name]
                etf_rows.loc[etf_rows.index[1:], 'ETF'] = ''
                sell_df = pd.concat([sell_df, etf_rows], ignore_index=True)
        return sell_df

    def _update_buy_recommendations(self, buy_df: pd.DataFrame, stock: str, processed_data: Dict) -> pd.DataFrame:
        lth = self._get_lifetime_high(
            self.secrets["connections"]["gsheets"]["worksheets"][stock])
        down_lb = round((processed_data['cmp'] - processed_data['last_buy_price']) /
                        processed_data['last_buy_price'] * 100, 2) if processed_data['last_buy_price'] != 0 else 0
        down_lth = round((lth - processed_data['cmp']) / lth * 100, 2)
        amount = 5000 if st.session_state.user in ('Amit', 'Deepti') else 2500
        qty = math.ceil(
            amount / processed_data['cmp']) if processed_data['cmp'] != 0 else 0
        
        print(stock, processed_data, down_lb, down_lth)

        if (down_lb <= DOWNSIDE_THRESHOLD and processed_data['pnl'] < 0  and processed_data['age'] > DURATION) or (
                processed_data['last_buy_price'] == 0 and down_lth >= LTH_DOWNSIDE_THRESHOLD):
            new_entry = pd.DataFrame({
                'ETF': [stock],
                'Down%': [round(processed_data['pnl'] * 100, 2)],
                "Down_LTH%": [down_lth],
                "LTH": [lth],
                'Down_LB%': [down_lb],
                'CMP': [processed_data['cmp']],
                'Amount': [amount],
                'Qty': [qty],
                'LB': [processed_data['last_buy_price']]
            })
            buy_df = pd.concat([buy_df, new_entry], ignore_index=True)
        return buy_df

    def _update_total_investment(self, investment_total: pd.DataFrame,
                                 total_invested: float, total_current_value: float) -> pd.DataFrame:
        roi = round(((total_current_value - total_invested) /
                    total_invested) * 100, 2) if total_invested != 0 else 0
        gain = round(total_current_value - total_invested, 2)
        return pd.concat([
            investment_total,
            pd.DataFrame({
                'Total Investment': [total_invested],
                'Current Value': [total_current_value],
                'ROI': [roi],
                'Gain': [gain]
            })
        ], ignore_index=True)

    def _display_summary(self, investment_total: pd.DataFrame) -> None:
        format_dict = {'Total Investment': '{:.2f}',
                       'Current Value': '{:.2f}', 'ROI': '{:.2f}', 'Gain': '{:.0f}'}
        styled_res = investment_total.round(2).style.format(
            format_dict).apply(self._highlight_gain_condition, axis=0)
        self.total_invested_place.dataframe(styled_res)

    def _display_individual_investments(self, investment_individual: pd.DataFrame) -> None:
        investment_individual = investment_individual.sort_values(
            "ROI", ascending=False).round(2)
        half = len(investment_individual) // 2
        res_individual_1 = investment_individual.iloc[:half]
        res_individual_2 = investment_individual.iloc[half:]

        format_dict = {'Total Investment': '{:.2f}', 'CMP': '{:.2f}', 'Buy Avg': '{:.2f}', 'Qty': '{:.2f}',
                       'Current Value': '{:.2f}', 'ROI': '{:.2f}', 'Gain': '{:.0f}'}

        styled_res_individual_1 = res_individual_1.style.format(format_dict).apply(
            self._highlight_gain_condition2, subset=['ROI'], axis=0)
        styled_res_individual_2 = res_individual_2.style.format(format_dict).apply(
            self._highlight_gain_condition2, subset=['ROI'], axis=0)

        num_rows = len(res_individual_1)
        self.col1.dataframe(styled_res_individual_1, use_container_width=True, height=(
            num_rows + 1) * 35 + 3)
        self.col2.dataframe(styled_res_individual_2, use_container_width=True, height=(
            num_rows + 2) * 35 + 3)

    def _display_buy_sell_recommendations(self, buy_df: pd.DataFrame, sell_df: pd.DataFrame) -> None:
        self.buy_head.subheader('Buy')
        self.buy_etf.dataframe(buy_df.sort_values(
            'Down_LB%'), use_container_width=True)

        self.sell_head.subheader('Sell')
        if not sell_df.empty:
            sell_df = sell_df.drop(columns=['Date'], axis=1, errors='ignore')
            numeric_cols = ['Price', 'Qty.', 'CMP', 'Gain%', 'Amount']
            for col in numeric_cols:
                if col in sell_df.columns:
                    sell_df[col] = pd.to_numeric(sell_df[col], errors='coerce')
            sell_df[numeric_cols] = sell_df[numeric_cols].fillna(0)
            format_dict = {col: '{:.2f}' for col in numeric_cols}
            styled_sell_df = sell_df.round(2).style.format(format_dict).apply(
                self._highlight_gain_condition3, subset=['Gain%'], axis=0)
            self.sell_etf.dataframe(styled_sell_df, use_container_width=True)

    def _get_cmp_price(self, ticker: str) -> float:
        url = f'https://www.google.com/finance/quote/{ticker}:NSE'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        price_element = soup.find(class_="YMlKec fxKbKc")
        return float(price_element.text[1:].replace(',', ''))

    def _get_lifetime_high(self, ticker):
        url = f'https://www.google.com/finance/quote/{ticker}:NSE'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        class1 = "P6K39c"
        index = soup.find_all(class_=class1)[2].text.index('-')
        return float(soup.find_all(class_=class1)[2].text[index + 3:].replace(',', ''))

    def _highlight_gain_condition3(self, s: pd.Series) -> pd.Series:
        if s.name == 'Gain%':
            return s.apply(self._highlight_gain_sell)
        return s

    def _highlight_gain_sell(self, x: float) -> str:
        if 3 < x <= 4:
            return 'background-color: rgba(255, 140, 0, 1)'
        elif x > 4:
            return 'background-color: rgba(63, 255, 0, 1)'
        return ''

    def _highlight_gain_condition(self, s: pd.Series) -> pd.Series:
        if s.name in ('ROI', 'Gain'):
            return s.apply(self._highlight_single_gain)
        elif s.name == 'Total Investment':
            return s.apply(self._highlight)
        return s.apply(self._highlight_2)

    def _highlight_gain_condition2(self, s: pd.Series) -> pd.Series:
        if s.name == 'ROI':
            return s.apply(self._highlight_roi)
        return s

    def _highlight_roi(self, value: float) -> str:
        if value < 0:
            return 'background-color: rgba(255, 0, 0, 0.8)'
        elif value == 0:
            return 'background-color: rgba(255, 192, 203, 0.7)'
        elif 0 < value <= 2:
            return 'background-color: rgba(255, 255, 0, 0.7)'
        elif 2 < value <= 3:
            return 'background-color: rgba(255, 140, 0, 1)'
        elif value > 3:
            return 'background-color: rgba(63, 255, 0, 1)'
        return ''

    def _highlight(self, x: float) -> str:
        return 'background-color: rgba(139,190,27,1)'

    def _highlight_2(self, x: float) -> str:
        return 'background-color: rgba(255, 140, 0, 1)'

    def _highlight_single_gain(self, value: float) -> str:
        if value <= 0:
            return 'background-color: rgba(255, 0, 0, 0.8)'
        return 'background-color: rgba(63, 255, 0, 1)'


# Main execution
if __name__ == "__main__":
    # st.set_page_config(
    #     page_title="ETFDash",
    #     page_icon="📈",
    #     layout="wide"
    # )
    dashboard = ETFDashboard()
    dashboard.run()
