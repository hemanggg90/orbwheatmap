from Dhan_Tradehull import Tradehull

tsl = Tradehull(
   ,
    mode="access_token"
)

symbols = [
    # paste ALL symbols from your sector configuration here
]

for symbol in symbols:
    symbol = symbol.strip().upper()

    try:
        CE_symbol_name, PE_symbol_name, strike = tsl.ATM_Strike_Selection(Underlying='NIFTY', Expiry=0)

        

        if CE_symbol_name is None or PE_symbol_name is None or strike is None:
            print(f"[EMPTY]  {symbol}")
        else:
            print(f"[OK]     {symbol} -> {len(CE_symbol_name)} (CE) and {len(PE_symbol_name)} (PE) candles")

    except Exception as e:
        print(f"[FAILED] {symbol} -> {type(e).__name__}: {e}")