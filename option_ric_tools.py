import refinitiv.dataplatform.eikon as ek
import pandas as pd
from datetime import timedelta
from datetime import datetime

ek.set_app_key('488246f60cb449c3a7edc9abedfb4d78cd678778')


def get_exchange_code(asset):
    # For simplicity, return OPQ for US equities like SPY
    # In a full implementation, this would use Eikon search API
    return ['OPQ']


def check_ric(ric, maturity):
    exp_date = pd.Timestamp(maturity)

    # get start and end date for get_timeseries query
    sdate = (datetime.now() - timedelta(90)).strftime('%Y-%m-%d')
    edate = datetime.now().strftime('%Y-%m-%d')

    # check if option is matured. If yes, recalculate start and end date
    if pd.Timestamp(maturity) < datetime.now():
        sdate = (exp_date - timedelta(90)).strftime('%Y-%m-%d')
        edate = exp_date.strftime('%Y-%m-%d')

    # request option prices using Eikon API
    try:
        prices = ek.get_timeseries(ric, start_date=sdate, end_date=edate, interval='daily', fields=['CLOSE'])
        if prices is not None and not prices.empty:
            prices.columns = ['TRDPRC_1']
            prices.index.name = 'Date'
            return ric, prices
    except:
        pass
    return ric, None


def get_exp_month(exp_date, opt_type):
    
    # define option expiration identifiers
    ident = {'1': {'exp': 'A','C': 'A', 'P': 'M'}, 
           '2': {'exp': 'B', 'C': 'B', 'P': 'N'}, 
           '3': {'exp': 'C', 'C': 'C', 'P': 'O'}, 
           '4': {'exp': 'D', 'C': 'D', 'P': 'P'},
           '5': {'exp': 'E', 'C': 'E', 'P': 'Q'},
           '6': {'exp': 'F', 'C': 'F', 'P': 'R'},
           '7': {'exp': 'G', 'C': 'G', 'P': 'S'}, 
           '8': {'exp': 'H', 'C': 'H', 'P': 'T'}, 
           '9': {'exp': 'I', 'C': 'I', 'P': 'U'}, 
           '10': {'exp': 'J', 'C': 'J', 'P': 'V'}, 
           '11': {'exp': 'K', 'C': 'K', 'P': 'W'}, 
           '12': {'exp': 'L', 'C': 'L', 'P': 'X'}}
    
    # get expiration month code for a month
    if opt_type.upper() == 'C':
        exp_month = ident[str(exp_date.month)]['C']
    elif opt_type.upper() == 'P':
        exp_month = ident[str(exp_date.month)]['P']
        
    return ident, exp_month


def get_ric_opra(asset, maturity, strike, opt_type):
    exp_date = pd.Timestamp(maturity)

    # trim underlying asset's RIC to get the required part for option RIC
    if asset[0] == '.': # check if the asset is an index or an equity
        asset_name = asset[1:] # get the asset name - we remove "." symbol for index options
    else:
        asset_name = asset.split('.')[0] # we need only the first part of the RICs for equities

    # Modern OPRA RIC format: Underlying + YYMMDD + C/P + strike*1000 (8 digits, zero-padded) + .U
    yy = str(exp_date.year)[-2:]
    mm = str(exp_date.month).zfill(2)
    dd = str(exp_date.day).zfill(2)
    cp = opt_type.upper()
    strike_8digit = str(int(strike * 1000)).zfill(8)

    ric = asset_name + yy + mm + dd + cp + strike_8digit + '.U'

    # For expired options, add the expiration syntax
    ident = {'1': {'exp': 'A'}, '2': {'exp': 'B'}, '3': {'exp': 'C'}, '4': {'exp': 'D'},
             '5': {'exp': 'E'}, '6': {'exp': 'F'}, '7': {'exp': 'G'}, '8': {'exp': 'H'},
             '9': {'exp': 'I'}, '10': {'exp': 'J'}, '11': {'exp': 'K'}, '12': {'exp': 'L'}}

    ric, prices = check_ric(ric, maturity)

    # return valid rics or append to the possible_ric list if no price is found
    possible_rics = []
    if prices is not None:
        return ric, prices
    else:
        possible_rics.append(ric)
        print(f'Here is a list of possible RICs {possible_rics}, however we could not find any prices for those!')
    return ric, prices


def get_ric_hk(asset, maturity, strike, opt_type):
    exp_date = pd.Timestamp(maturity)
    
    # get asset name and strike price for the asset
    if asset[0] == '.': 
        asset_name = asset[1:] 
        strike_ric = str(int(strike))
    else:
        asset_name = asset.split('.')[0]
        strike_ric = str(int(strike * 100))
     
    # get expiration month codes
    ident, exp_month = get_exp_month(exp_date, opt_type)
 
    possible_rics = []
    # get rics for options on indexes. Return if valid add to the possible_rics list if no price is found
    if asset[0] == '.':
        ric = asset_name + strike_ric + exp_month + str(exp_date.year)[-1:] + '.HF'
        ric, prices = check_ric(ric, maturity)
        if prices is not None:
            return ric, prices
        else:
            possible_rics.append(ric)
    else:
        # get rics for options on equities. Return if valid add to the possible_rics list if no price is found
        # there could be several generations of options depending on the number of price adjustments due to a corporate event
        # here we use 4 adjustment opportunities.
        for i in range(4):
            ric = asset_name + strike_ric + str(i)+ exp_month + str(exp_date.year)[-1:] + '.HK'
            ric, prices = check_ric(ric, maturity)
            if prices is not None:
                 return ric, prices # we return ric and prices for the first found ric (we don't check for other adjusted rics)
            else:
                possible_rics.append(ric)
    print(f'Here is a list of possible RICs {possible_rics}, however we could not find any prices for those!')
    return  ric, prices


def get_ric_ose(asset, maturity, strike, opt_type):
    exp_date = pd.Timestamp(maturity)
 
    if asset[0] == '.':
        asset_name = asset[1:]
    else:
        asset_name = asset.split('.')[0]
    strike_ric = str(strike)[:3]
        
    ident, exp_month = get_exp_month(exp_date, opt_type)
    
    possible_rics = []
    if asset[0] == '.':
        # Option Root codes for indexes are different from the RIC, so we rename where necessery
        if asset_name == 'N225':
            asset_name = 'JNI'
        elif asset_name == 'TOPX':
            asset_name = 'JTI'
        # we consider also J-NET (Off-Auction(with "L")) and High  frequency (with 'R') option structures 
        for jnet in ['', 'L', 'R']:
            ric = asset_name + jnet + strike_ric + exp_month + str(exp_date.year)[-1:] + '.OS'
            ric, prices = check_ric(ric, maturity)
            if prices is not None:
                return ric, prices
            else:
                possible_rics.append(ric)
    else:
        generations = ['Y', 'Z', 'A', 'B', 'C'] # these are generation codes similar to one from HK 
        for jnet in ['', 'L', 'R']:
            for gen in generations:
                ric = asset_name + jnet + gen + strike_ric + exp_month + str(exp_date.year)[-1:] + '.OS'
                ric, prices = check_ric(ric, maturity)
                if prices is not None:
                    return ric, prices
                else:
                    possible_rics.append(ric)
    print(f'Here is a list of possible RICs {possible_rics}, however we could not find any prices for those!')
    return  ric, prices


def get_ric_eurex(asset, maturity, strike, opt_type):
    exp_date = pd.Timestamp(maturity)
 
    if asset[0] == '.': 
        asset_name = asset[1:]
        if asset_name == 'FTSE':
            asset_name = 'OTUK'
        elif asset_name == 'SSMI':
            asset_name = 'OSMI'
        elif asset_name == 'GDAXI':
            asset_name = 'GDAX'
        elif asset_name == 'ATX':
            asset_name = 'FATXA'
        elif asset_name == 'STOXX50E':
            asset_name = 'STXE'           
    else:
        asset_name = asset.split('.')[0]
        
    ident, exp_month = get_exp_month(exp_date, opt_type)
        
    if type(strike) == float:
        int_part = int(strike)
        dec_part = str(str(strike).split('.')[1])[0]
    else:
        int_part = int(strike)
        dec_part = '0'      
        
    if len(str(int(strike))) == 1:
        strike_ric = '0' + str(int_part) + dec_part
    else:
        strike_ric = str(int_part) + dec_part
    
    possible_rics = []
    generations = ['', 'a', 'b', 'c', 'd']
    for gen in generations:
        ric = asset_name + strike_ric  + gen + exp_month + str(exp_date.year)[-1:] + '.EX'
        ric, prices = check_ric(ric, maturity)
        if prices is not None:
            return ric, prices
        else:
            possible_rics.append(ric)
    print(f'Here is a list of possible RICs {possible_rics}, however we could not find any prices for those!')
    return  ric, prices


def get_ric_ieu(asset, maturity, strike, opt_type):
    exp_date = pd.Timestamp(maturity)
    
    if asset[0] == '.':
        asset_name = asset[1:]
        if asset_name == 'FTSE':
            asset_name = 'LFE'       
    else:
        asset_name = asset.split('.')[0] 
        
    ident, exp_month = get_exp_month(exp_date, opt_type)
 
    if len(str(int(strike))) == 2:
        strike_ric = '0' + str(int(strike))
    else:
        strike_ric = str(int(strike))
        
    if type(strike) == float and len(str(int(strike))) == 1:
        int_part = int(strike)
        dec_part = str(str(strike).split('.')[1])[0]        
        strike_ric = '0' + str(int_part) + dec_part
    
    possible_rics = []
    generations = ['', 'a', 'b', 'c', 'd']
    for gen in generations:
        ric = asset_name + strike_ric  + gen + exp_month + str(exp_date.year)[-1:] + '.L'
        ric, prices = check_ric(ric, maturity)
        if prices is not None:
            return ric, prices
        else:
            possible_rics.append(ric)
    print(f'Here is a list of possible RICs {possible_rics}, however we could not find any prices for those!')
    return  ric, prices


def get_optionRic(isin, maturity, strike, opt_type):
    
    # define covered exchanges along with functions to get RICs from
    exchanges = {'OPQ': get_ric_opra,
           'IEU': get_ric_ieu,
           'EUX': get_ric_eurex,
           'HKG': get_ric_hk,
           'HFE': get_ric_hk,
           'OSA': get_ric_ose}
    
    # convert ISIN to RIC
    try:
        ricUnderlying = ek.get_symbology(isin, from_symbol_type='ISIN', to_symbol_type='RIC')['RIC'][0]
    except:
        # For SPY special case
        if isin == 'US78462F1030':
            ricUnderlying = 'SPY'
        else:
            raise ValueError(f"Could not convert ISIN {isin} to RIC")
    
    # get exchanges codes where the option on the given asset is traded
    exchnage_codes = get_exchange_code(ricUnderlying)
    
    # get the list of (from all available and covered exchanges) valid rics and their prices
    option_rics = [] 
    priceslist = []
    for exch in exchnage_codes:
        if exch in exchanges.keys():
            ric, prices = exchanges[exch](ricUnderlying, maturity, strike, opt_type)
            if prices is not None:
                option_rics.append(ric)
                priceslist.append(prices)
                print(f'Option RIC for {exch} exchange is successfully constructed')     
        else:
            print(f'The {exch} exchange is not supported yet')
    return option_rics, priceslist
