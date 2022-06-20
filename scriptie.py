import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from statistics import mean
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import scipy
import numpy as np
from statsmodels.compat import lzip
import statsmodels.stats.api as sms


def regress(x,constant,rmfi,smbi,hmli,voli,momi,rdi,envi,capexi,pmbi,pmb2i,imni,imn2i):

    rd = pd.read_excel('esgdata.xlsx',
                          sheet_name='return2', header=0)
    vr = pd.read_excel('esgdata.xlsx',
                          sheet_name='variables', header=0)


    ys = (rd.columns)
    start = 104
    coefficients = []
    standarderrors = []
    returns = []
    returnspol = []
    returnsnpol = []

    while start < 431:

        begin = start - 104
        X = vr[x].iloc[begin:start]
        X = sm.add_constant(X)
        cons = []
        pol = []
        models = pd.DataFrame()
        for i in range(len(ys)):
            Y = rd[ys[i]]
            Y = Y[begin:start]
            model = sm.OLS(Y, X).fit(cov_type='HC3')
            coefficients.append(model.params)
            cons.append(model.params['const'])
            standarderrors.append(model.HC3_se)
            pol.append(rd[ys[i]].iloc[431])

        models.insert(0,'CUSIP',ys)
        models.insert(1,'const',cons)
        models.insert(2,'poluting',pol)
        modelspol = models.loc[models['poluting'] == 1]
        modelsnpol = models.loc[models['poluting'] == 0]

        models = models.sort_values(by='const', ascending=False)
        modelspol = modelspol.sort_values(by='const', ascending=False)
        modelsnpol = modelsnpol.sort_values(by='const', ascending=False)
        length132 = round(0.1 * len(modelspol))
        length133 = round(0.1 * len(modelsnpol))
        length13 = round(0.1 * len(models))
        c = 0
        d = 0
        e = 0
        if (start+13)<430:
            end = (start+13)
        else:
            end = 430
        for i in range(start,end):
            for j in range(length13):
                c = c + (rd.iloc[i, rd.columns.get_loc(models.iloc[j, models.columns.get_loc('CUSIP')])])
            for j in range(length132):
                d = d + (rd.iloc[i, rd.columns.get_loc(modelspol.iloc[j, modelspol.columns.get_loc('CUSIP')])])
            for j in range(length133):
                e = e + (rd.iloc[i, rd.columns.get_loc(modelsnpol.iloc[j, modelsnpol.columns.get_loc('CUSIP')])])

        c = c/(length13)
        d = d/length132
        e = e/length132
        returns.append(c)
        returnspol.append(d)
        returnsnpol.append(e)
        start = start + 13

    coef = pd.DataFrame(coefficients)
    stde = pd.DataFrame(standarderrors)
    constant.append(coef['const'].mean())
    rmfi.append(coef['rmf'].mean())
    smbi.append(coef['smb'].mean())
    hmli.append(coef['hml'].mean())
    voli.append(coef['vol'].mean())
    momi.append(coef['mom'].mean())
    if len(x) == 6:
        if x[5] == 'rd':
            rdi.append(coef['rd'].mean())
        elif x[5] == 'env':
            envi.append(coef['env'].mean())
        elif x[5] == 'capexi':
            capexi.append(coef['capex'].mean())
        elif x[5] == 'pmb':
            pmbi.append(coef['pmb'].mean())
        elif x[5] == 'pmb2':
            pmb2i.append(coef['pmb2'].mean())
        elif x[5] == 'imn':
            imni.append(coef['imn'].mean())
        elif x[5] == 'imn2':
            imn2i.append(coef['imn2'].mean())


    years = (7+1/3)
    yearr = sum(returns)/years
    stdevr = (np.std(returns))
    yearrp = sum(returnspol)/years
    stdevrp = (np.std(returnspol))
    yearrnp = sum(returnsnpol)/years
    stdevrnp = (np.std(returnsnpol))
    print(yearr)
    print(yearrp)
    print(yearrnp)
    print(yearr/stdevr)
    print(yearrp / stdevrp)
    print(yearrnp / stdevrnp)

    return constant,rmfi,smbi,hmli,voli,momi,rdi,envi,capexi,pmbi,pmb2i,imni,imn2i








def vol(month,volatillity):


    rd = pd.read_excel('esgdata.xlsx',
                           sheet_name='return', header=0)


    vol = pd.read_excel('esgdata.xlsx',
                 sheet_name='volatiliity', header=0)
    sz= pd.read_excel('esgdata.xlsx',
                           sheet_name='size', header=0)

    rd['month_year'] = pd.to_datetime(rd['date']).dt.to_period('M')
    vol['month_year'] = pd.to_datetime(vol['dates']).dt.to_period('M')
    vol['year'] = pd.DatetimeIndex(vol['dates']).year
    vol1  = vol.iloc[month]
    my = vol1['month_year']
    year  = vol1['year']-1
    smb13 = sz.loc[sz['Assessment Year'] == year]
    smb13 = smb13.sort_values(by='Market Value - Total - Fiscal', ascending=True)

    length13 = round(0.1*len(smb13))
    b13 = (len(smb13)- length13)
    datas = []
    datab = []
    for row in range(length13):
        datas.append(smb13['CUSIP'].iloc[row])
        datab.append(smb13['CUSIP'].iloc[b13+row])

    volas= []
    volab = []
    for values in datas:
        volas.append((vol1[values]))
    for values in datab:
        volab.append((vol1[values]))
    volb = pd.DataFrame()
    vols = pd.DataFrame()
    vols.insert(0,'CUSIP',datas)
    vols.insert(1, 'volatillity', volas)
    volb.insert(0,'CUSIP',datab)
    volb.insert(1, 'volatillity',  volab)

    vols =  vols.sort_values(by='volatillity', ascending=True)
    volb = volb.sort_values(by='volatillity', ascending=True)
    length13 = round(0.5*len(vols))
    b13 = (len(vols)- round(0.2*len(vols)))

    datasv = []
    datass = []
    for row in range(length13):
        datass.append(vols.iloc[row])
    for row in range(len(vols)-b13):
        datasv.append(vols.iloc[b13+row])



    length132 = round(0.2*len(volb))
    b132 = (len(volb)- round(0.5*len(volb)))
    databs = []
    databv = []


    for row in range(length132):
        databs.append(volb.iloc[row])

    for row in range(len(volb)-b132):
        databv.append(volb.iloc[b132 + row])

    volss = pd.DataFrame(datass)
    volsv = pd.DataFrame(datasv)
    volbs = pd.DataFrame(databs)
    volbv = pd.DataFrame(databv)


    length13 = round(len(volss))
    b13 = len(volsv)
    length132 = round(len(volbs))
    b132 = len(volbv)
    rd = rd.loc[rd['month_year']== my]

    for i in range(len(rd)):
        c = 0
        d = 0
        e = 0
        f = 0
        a  = 0
        for j in range(length13):
            c = c+ (rd.iloc[i, rd.columns.get_loc(volss.iloc[j, volss.columns.get_loc('CUSIP')])])
        for j in range(b13):
            e = e +(rd.iloc[i, rd.columns.get_loc(volsv.iloc[j, volsv.columns.get_loc('CUSIP')])])

        for j in range(length132):
            d= d +(rd.iloc[i, rd.columns.get_loc(volbs.iloc[j, volbs.columns.get_loc('CUSIP')])])
        for j in range(b132):
            f = f+(rd.iloc[i, rd.columns.get_loc(volbv.iloc[j, volbv.columns.get_loc('CUSIP')])])
        a = (1/2*((c/length13)-(e/b13)))+(1/2*((d/length132)-(f/b132)))
        volatillity.append(a)
    return volatillity

def mom(month,momentum):



    rd = pd.read_excel('esgdata.xlsx',
                           sheet_name='return', header=0)


    mom = pd.read_excel('esgdata.xlsx',
                 sheet_name='mom2', header=0)
    sz= pd.read_excel('esgdata.xlsx',
                           sheet_name='size', header=0)

    rd['month_year'] = pd.to_datetime(rd['date']).dt.to_period('M')
    mom['month_year'] = pd.to_datetime(mom['periods']).dt.to_period('M')
    mom['year'] = pd.DatetimeIndex(mom['periods']).year
    mom1  = mom.iloc[month]
    my = mom1['month_year']
    year  = mom1['year']-1



    smb13 = sz.loc[sz['Assessment Year'] == year]
    smb13 = smb13.sort_values(by='Market Value - Total - Fiscal', ascending=True)



    length13 = round(0.1*len(smb13))
    b13 = (len(smb13)- length13)
    datas = []
    datab = []
    for row in range(length13):
        datas.append(smb13['CUSIP'].iloc[row])
        datab.append(smb13['CUSIP'].iloc[b13+row])


    momentums= []
    momentumb = []
    for values in datas:
        momentums.append((mom1[values]))

    for values in datab:
        momentumb.append(mom1[values])

    momb = pd.DataFrame()
    moms = pd.DataFrame()
    moms.insert(0,'CUSIP',datas)
    moms.insert(1, 'momentum', momentums)
    momb.insert(0,'CUSIP',datab)
    momb.insert(1, 'momentum', momentumb)


    moms = moms.sort_values(by='momentum', ascending=True)
    momb = momb.sort_values(by='momentum', ascending=True)

    length13 = round(0.3*len(moms))
    b13 = (len(moms)- length13)
    datash = []
    datasl = []
    databl = []
    databh = []

    for row in range(length13):
        datasl.append(moms.iloc[row])
        datash.append(moms.iloc[b13+row])
        databl.append(momb.iloc[row])
        databh.append(momb.iloc[b13 + row])

    momsl = pd.DataFrame(datasl)
    momsw = pd.DataFrame(datash)
    mombl = pd.DataFrame(databl)
    mombw = pd.DataFrame(databh)
    length13 = round(len(momsw))
    b13 = (len(momsw) - length13)
    rd = rd.loc[rd['month_year']== my]
    for i in range (len(rd)):
        c = 0
        d = 0
        e = 0
        f = 0
        a  = 0
        for j in range(length13):
            c = c+ (rd.iloc[i, rd.columns.get_loc(momsw.iloc[j, momsw.columns.get_loc('CUSIP')])])
            e = e +(rd.iloc[i, rd.columns.get_loc(momsl.iloc[(b13 + j), momsl.columns.get_loc('CUSIP')])])

            d= d +(rd.iloc[i, rd.columns.get_loc(mombw.iloc[j, mombw.columns.get_loc('CUSIP')])])
            f = f+(rd.iloc[i, rd.columns.get_loc(mombl.iloc[(b13+ j), mombl.columns.get_loc('CUSIP')])])
        a = ((1/2)*1/length13*(c-e))+((1/2)*1/length13*(d-f))
        momentum.append(a)
    return momentum


def smbv(year,smb):
    if year == 2013:
        period1 = 1
        period2 = 53
    elif year == 2014:
        period1 = 53
        period2 = 106
    elif year == 2015:
        period1 = 106
        period2 = 158
    elif year == 2016:
        period1 = 158
        period2 = 210
    elif year == 2017:
        period1 = 210
        period2 = 262
    elif year == 2018:
        period1 = 262
        period2 = 314
    elif year == 2019:
        period1 = 314
        period2 = 367
    elif year == 2020:
        period1 = 367
        period2 = 419
    elif year == 2021:
        period1 = 419
        period2 = 432
    else:
        print('error')


    rd = pd.read_excel('esgdata.xlsx',
                 sheet_name='return', header=0)
    sz= pd.read_excel('esgdata.xlsx',
                           sheet_name='smb', header=0)


    smb13 = sz.loc[sz['Assessment Year'] == year]
    smb13 = smb13.sort_values(by='Market Value - Total - Fiscal', ascending=True)
    length13 = round(0.1*len(smb13))
    b13 = (len(smb13)- length13)

    for i in range (period1,period2):
        a = 0
        c = 0
        d = 0
        for j in range(length13):
            c = c+(rd.iloc[i, rd.columns.get_loc(smb13.iloc[j, smb13.columns.get_loc('CUSIP')])])
            d= d+ (rd.iloc[i, rd.columns.get_loc(smb13.iloc[(b13+j), smb13.columns.get_loc('CUSIP')])])

        a = ((1/3)*(1/length13)*c)-((1/3)*(1/length13)*d)
        smb.append(a)

    return smb

def hmlv(year,hml):
    if year == 2013:
        period1 = 1
        period2 = 53
    elif year == 2014:
        period1 = 53
        period2 = 106
    elif year == 2015:
        period1 = 106
        period2 = 158
    elif year == 2016:
        period1 = 158
        period2 = 210
    elif year == 2017:
        period1 = 210
        period2 = 262
    elif year == 2018:
        period1 = 262
        period2 = 314
    elif year == 2019:
        period1 = 314
        period2 = 367
    elif year == 2020:
        period1 = 367
        period2 = 419
    elif year == 2021:
        period1 = 419
        period2 = 432
    else:
        print('error')

    rd = pd.read_excel('esgdata.xlsx',
                 sheet_name='return', header=0)
    hl= pd.read_excel('esgdata.xlsx',
                           sheet_name='hml', header=0)



    # 13
    hml13 = hl.loc[hl['Assessment Year'] == year]
    hml13 = hml13.sort_values(['Market Value - Total - Fiscal'], ascending=[True])

    length13 = round(0.1*len(hml13))
    b13 = (len(hml13)- length13)

    datas = []
    datab = []
    for row in range(length13):
        datas.append(hml13.iloc[row])
        datab.append(hml13.iloc[b13+row])

    hmls = pd.DataFrame(datas)
    hmlb = pd.DataFrame(datab)

    hmls = hmls.sort_values(['bookvalue'], ascending=[True])
    hmlb = hmlb.sort_values(['bookvalue'], ascending=[True])


    length13 = round(0.3*len(hmls))
    b13 = (len(hmls)- length13)
    datasg = []
    datasv = []
    databg = []
    databv = []

    for row in range(length13):
        datasg.append(hmls.iloc[row])
        datasv.append(hmls.iloc[b13+row])
        databg.append(hmlb.iloc[row])
        databv.append(hmlb.iloc[b13+row])

    hmlsg = pd.DataFrame(datasg)
    hmlsv = pd.DataFrame(datasv)
    hmlbg = pd.DataFrame(databg)
    hmlbv = pd.DataFrame(databv)

    length13 = round(len(hmlsg))
    b13 = (len(hmlsg)- length13)

    for i in range (period1,period2):
        a = 0
        c = 0
        d = 0
        e = 0
        f = 0
        for j in range(length13):
            c = c+ (rd.iloc[i, rd.columns.get_loc(hmlsv.iloc[j, hmlsv.columns.get_loc('CUSIP')])])
            e = e +(rd.iloc[i, rd.columns.get_loc(hmlsg.iloc[(b13+j), hmlsg.columns.get_loc('CUSIP')])])

            d= d +(rd.iloc[i, rd.columns.get_loc(hmlbv.iloc[j, hmlbv.columns.get_loc('CUSIP')])])
            f = f+(rd.iloc[i, rd.columns.get_loc(hmlbg.iloc[(b13 + j), hmlbg.columns.get_loc('CUSIP')])])
        a = ((1/2)*(1/length13)*(c-e))+((1/2)*(1/length13)*(d-f))
        hml.append(a)


    return hml

# size rd esg
def esgv(year,esg):
    if year == 2013:
        period1 = 1
        period2 = 53
    elif year == 2014:
        period1 = 53
        period2 = 106
    elif year == 2015:
        period1 = 106
        period2 = 158
    elif year == 2016:
        period1 = 158
        period2 = 210
    elif year == 2017:
        period1 = 210
        period2 = 262
    elif year == 2018:
        period1 = 262
        period2 = 314
    elif year == 2019:
        period1 = 314
        period2 = 367
    elif year == 2020:
        period1 = 367
        period2 = 419
    elif year == 2021:
        period1 = 419
        period2 = 432
    else:
        print('error')

    rd = pd.read_excel('esgdata.xlsx',
                           sheet_name='return', header=0)
    es= pd.read_excel('esgdata.xlsx',
                           sheet_name='pmb', header=0)


    # 13
    esg13 = es.loc[es['Assessment Year'] == year]
    esg13 = esg13.sort_values(['Market Value - Total - Fiscal'], ascending=[True])

    length13 = round(0.5*len(esg13))
    b13 = (len(esg13)- length13)
    datas = []
    datab = []
    for row in range(length13):
        datas.append(esg13.iloc[row])
        datab.append(esg13.iloc[b13+row])

    esgs = pd.DataFrame(datas)
    esgb = pd.DataFrame(datab)

    esgs = esgs.sort_values(['Research and Development Expense'], ascending=[True])
    esgb = esgb.sort_values(['Research and Development Expense'], ascending=[True])

    length13 = round(0.3*len(esgs))
    b13 = (len(esgs)- length13)
    datash = []
    datasl = []
    databh = []
    databl = []

    for row in range(length13):
        datasl.append(esgs.iloc[row])
        datash.append(esgs.iloc[b13+row])
        databl.append(esgb.iloc[row])
        databh.append(esgb.iloc[b13+row])

    esgsl = pd.DataFrame(datasl)
    esgsh = pd.DataFrame(datash)
    esgbl = pd.DataFrame(databl)
    esgbh = pd.DataFrame(databh)

    esgsl = esgsl.sort_values(['Score Value env'], ascending=[True])
    esgbl = esgbl.sort_values(['Score Value env'], ascending=[True])
    esgsh = esgsh.sort_values(['Score Value env'], ascending=[True])
    esgbh = esgbh.sort_values(['Score Value env'], ascending=[True])

    length13 = round(0.3*len(esgsh))
    b13 = (len(esgsh)- length13)
    datashg = []
    datashv = []
    dataslg = []
    dataslv = []
    databhg = []
    databhv = []
    datablg = []
    datablv = []


    for row in range(length13):
        dataslg.append(esgsl.iloc[row])
        datablg.append(esgbl.iloc[row])
        datashg.append(esgsh.iloc[row])
        databhg.append(esgbh.iloc[row])
        dataslv.append(esgsl.iloc[b13+row])
        datablv.append(esgbl.iloc[b13+row])
        datashv.append(esgsh.iloc[b13+row])
        databhv.append(esgbh.iloc[b13+row])
    esgslg = pd.DataFrame(dataslg)
    esgshg = pd.DataFrame(datashg)
    esgslv = pd.DataFrame(dataslv)
    esgshv = pd.DataFrame(datashv)
    esgblg = pd.DataFrame(datablg)
    esgbhg = pd.DataFrame(databhg)
    esgblv = pd.DataFrame(datablv)
    esgbhv =pd.DataFrame(databhv)
    length13 = round(len(esgshv))
    b13 = (len(esgshv)- length13)
    for i in range (period1,period2):
        c = 0
        d = 0
        e = 0
        f = 0
        a  = 0
        for j in range(length13):
            c = c+ (rd.iloc[i, rd.columns.get_loc(esgshv.iloc[j, esgshv.columns.get_loc('CUSIP')])])
            e = e +(rd.iloc[i, rd.columns.get_loc(esgslg.iloc[(b13+j), esgslg.columns.get_loc('CUSIP')])])

            d= d +(rd.iloc[i, rd.columns.get_loc(esgbhv.iloc[j, esgbhv.columns.get_loc('CUSIP')])])
            f = f+(rd.iloc[i, rd.columns.get_loc(esgblg.iloc[(b13 + j), esgblg.columns.get_loc('CUSIP')])])
        a = ((1/2)*(1/length13)*(c-e))+((1/2)*(1/length13)*(d-f))
        esg.append(a)

    return esg
# esg
def esgv2(year,esg2):
    if year == 2013:
        period1 = 1
        period2 = 53
    elif year == 2014:
        period1 = 53
        period2 = 106
    elif year == 2015:
        period1 = 106
        period2 = 158
    elif year == 2016:
        period1 = 158
        period2 = 210
    elif year == 2017:
        period1 = 210
        period2 = 262
    elif year == 2018:
        period1 = 262
        period2 = 314
    elif year == 2019:
        period1 = 314
        period2 = 367
    elif year == 2020:
        period1 = 367
        period2 = 419
    elif year == 2021:
        period1 = 419
        period2 = 432
    else:
        print('error')

    rd = pd.read_excel('esgdata.xlsx',
                           sheet_name='return', header=0)
    es= pd.read_excel('esgdata.xlsx',
                           sheet_name='esg', header=0)


    # 13
    esg13 = es.loc[es['Assessment Year'] == year]
    esg13 = esg13.sort_values(['Market Value - Total - Fiscal'], ascending=[True])

    length13 = round(0.3*len(esg13))
    b13 = (len(esg13)- length13)
    datas = []
    datab = []
    for row in range(length13):
        datas.append(esg13.iloc[row])
        datab.append(esg13.iloc[b13+row])

    esgs = pd.DataFrame(datas)
    esgb = pd.DataFrame(datab)

    esgs = esgs.sort_values(['Score Value env'], ascending=[True])
    esgb = esgb.sort_values(['Score Value env'], ascending=[True])

    length13 = round(0.3*len(esgs))
    b13 = (len(esgs)- length13)
    datasg = []
    datasv = []
    databg = []
    databv = []

    for row in range(length13):
        datasg.append(esgs.iloc[row])
        datasv.append(esgs.iloc[b13+row])
        databg.append(esgb.iloc[row])
        databv.append(esgb.iloc[b13+row])

    esgsg = pd.DataFrame(datasg)
    esgsv = pd.DataFrame(datasv)
    esgbg = pd.DataFrame(databg)
    esgbv = pd.DataFrame(databv)

    length13 = round(len(esgsg))
    b13 = (len(esgsg)- length13)
    for i in range (period1,period2):
        c = 0
        d = 0
        e = 0
        f = 0
        for j in range(length13):
            c = c+ (rd.iloc[i, rd.columns.get_loc(esgsv.iloc[j, esgsv.columns.get_loc('CUSIP')])])
            e = e +(rd.iloc[i, rd.columns.get_loc(esgsg.iloc[(b13+j), esgsg.columns.get_loc('CUSIP')])])

            d= d +(rd.iloc[i, rd.columns.get_loc(esgbv.iloc[j, esgbv.columns.get_loc('CUSIP')])])
            f = f+(rd.iloc[i, rd.columns.get_loc(esgbg.iloc[(b13 + j), esgbg.columns.get_loc('CUSIP')])])
        a = ((1/2)*(1/length13)*(c-e))+((1/2)*(1/length13)*(d-f))
        esg2.append(a)
    return esg2
# size esg rd
def esgv3(year,esg3):
    if year == 2013:
        period1 = 1
        period2 = 53
    elif year == 2014:
        period1 = 53
        period2 = 106
    elif year == 2015:
        period1 = 106
        period2 = 158
    elif year == 2016:
        period1 = 158
        period2 = 210
    elif year == 2017:
        period1 = 210
        period2 = 262
    elif year == 2018:
        period1 = 262
        period2 = 314
    elif year == 2019:
        period1 = 314
        period2 = 367
    elif year == 2020:
        period1 = 367
        period2 = 419
    elif year == 2021:
        period1 = 419
        period2 = 432
    else:
        print('error')

    rd = pd.read_excel('esgdata.xlsx',
                           sheet_name='return', header=0)
    es= pd.read_excel('esgdata.xlsx',
                           sheet_name='pmb', header=0)


    # 13
    esg13 = es.loc[es['Assessment Year'] == year]
    esg13 = esg13.loc[es['poluting'] == 1]
    esg13 = esg13.sort_values(['Market Value - Total - Fiscal'], ascending=[True])

    length13 = round(0.5*len(esg13))
    b13 = (len(esg13)- length13)
    datas = []
    datab = []
    for row in range(length13):
        datas.append(esg13.iloc[row])
        datab.append(esg13.iloc[b13+row])

    esgs = pd.DataFrame(datas)
    esgb = pd.DataFrame(datab)

    esgs = esgs.sort_values(['Score Value env'], ascending=[True])
    esgb = esgb.sort_values(['Score Value env'], ascending=[True])

    length13 = round(0.3*len(esgs))
    b13 = (len(esgs)- length13)
    datash = []
    datasl = []
    databh = []
    databl = []

    for row in range(length13):
        datasl.append(esgs.iloc[row])
        datash.append(esgs.iloc[b13+row])
        databl.append(esgb.iloc[row])
        databh.append(esgb.iloc[b13+row])

    esgsl = pd.DataFrame(datasl)
    esgsh = pd.DataFrame(datash)
    esgbl = pd.DataFrame(databl)
    esgbh = pd.DataFrame(databh)

    esgsl = esgsl.sort_values(['Research and Development Expense'], ascending=[True])
    esgbl = esgbl.sort_values(['Research and Development Expense'], ascending=[True])
    esgsh = esgsh.sort_values(['Research and Development Expense'], ascending=[True])
    esgbh = esgbh.sort_values(['Research and Development Expense'], ascending=[True])
    length13 = round(0.3*len(esgsh))
    b13 = (len(esgsh)- length13)
    datashg = []
    datashv = []
    dataslg = []
    dataslv = []
    databhg = []
    databhv = []
    datablg = []
    datablv = []


    for row in range(length13):
        dataslg.append(esgsl.iloc[row])
        datablg.append(esgbl.iloc[row])
        datashg.append(esgsh.iloc[row])
        databhg.append(esgbh.iloc[row])
        dataslv.append(esgsl.iloc[b13+row])
        datablv.append(esgbl.iloc[b13+row])
        datashv.append(esgsh.iloc[b13+row])
        databhv.append(esgbh.iloc[b13+row])
    esgslg = pd.DataFrame(dataslg)
    esgshg = pd.DataFrame(datashg)
    esgslv = pd.DataFrame(dataslv)
    esgshv = pd.DataFrame(datashv)
    esgblg = pd.DataFrame(datablg)
    esgbhg = pd.DataFrame(databhg)
    esgblv = pd.DataFrame(datablv)
    esgbhv =pd.DataFrame(databhv)

    length13 = round(len(esgshv))
    b13 = (len(esgshv)- length13)

    for i in range (period1,period2):
        c = 0
        d = 0
        e = 0
        f = 0
        a = 0
        for j in range(length13):
            c = c+ (rd.iloc[i, rd.columns.get_loc(esgshv.iloc[j, esgshv.columns.get_loc('CUSIP')])])
            e = e +(rd.iloc[i, rd.columns.get_loc(esgslg.iloc[(b13+j), esgslg.columns.get_loc('CUSIP')])])

            d= d +(rd.iloc[i, rd.columns.get_loc(esgbhv.iloc[j, esgbhv.columns.get_loc('CUSIP')])])
            f = f+(rd.iloc[i, rd.columns.get_loc(esgblg.iloc[(b13 + j), esgblg.columns.get_loc('CUSIP')])])
        a = ((1/2)*(1/length13)*(c-e))+((1/2)*(1/length13)*(d-f))
        esg3.append(a)

    return esg3


def rdv2(year, rd2):
    if year == 2013:
        period1 = 1
        period2 = 53
    elif year == 2014:
        period1 = 53
        period2 = 106
    elif year == 2015:
        period1 = 106
        period2 = 158
    elif year == 2016:
        period1 = 158
        period2 = 210
    elif year == 2017:
        period1 = 210
        period2 = 262
    elif year == 2018:
        period1 = 262
        period2 = 314
    elif year == 2019:
         period1 = 314
         period2 = 367
    elif year == 2020:
        period1 = 367
        period2 = 419
    elif year == 2021:
        period1 = 419
        period2 = 432
    else:
        print('error')

    rd = pd.read_excel('esgdata.xlsx',
                        sheet_name='return', header=0)
    capex13 = pd.read_excel('esgdata.xlsx',
                        sheet_name='rd', header=0)

        # 13
    capex13 = capex13.loc[capex13['Assessment Year'] == year]
    capex13 = capex13.sort_values(['Market Value - Total - Fiscal'], ascending=[True])

    length13 = round(0.1 * len(capex13))
    b13 = (len(capex13) - length13)
    datas = []
    datab = []
    for row in range(length13):
        datas.append(capex13.iloc[row])
        datab.append(capex13.iloc[b13 + row])

    caps = pd.DataFrame(datas)
    capb = pd.DataFrame(datab)

    caps = caps.sort_values(['Research and Development Expense'], ascending=[True])
    capb = capb.sort_values(['Research and Development Expense'], ascending=[True])

    length13 = round(0.3 * len(caps))
    b13 = (len(caps) - length13)
    datasg = []
    datasv = []
    databg = []
    databv = []

    for row in range(length13):
        datasg.append(caps.iloc[row])
        datasv.append(caps.iloc[b13 + row])
        databg.append(capb.iloc[row])
        databv.append(capb.iloc[b13 + row])

    capsg = pd.DataFrame(datasg)
    capsv = pd.DataFrame(datasv)
    capbg = pd.DataFrame(databg)
    capbv = pd.DataFrame(databv)

    length13 = round(len(capsg))
    b13 = (len(capsg) - length13)
    for i in range(period1, period2):
        c = 0
        d = 0
        e = 0
        f = 0
        for j in range(length13):
            c = c + (rd.iloc[i, rd.columns.get_loc(capsv.iloc[j, capsv.columns.get_loc('CUSIP')])])
            e = e + (rd.iloc[i, rd.columns.get_loc(capsg.iloc[(b13 + j), capsg.columns.get_loc('CUSIP')])])

            d = d + (rd.iloc[i, rd.columns.get_loc(capbv.iloc[j, capbv.columns.get_loc('CUSIP')])])
            f = f + (rd.iloc[i, rd.columns.get_loc(capbg.iloc[(b13 + j), capbg.columns.get_loc('CUSIP')])])
        a = ((1 / 2) * (1 / length13) * (c - e)) + ((1 / 2) * (1 / length13) * (d - f))
        rd2.append(a)
    return rd2

def capexv(year, capex):
    if year == 2013:
        period1 = 1
        period2 = 53
    elif year == 2014:
        period1 = 53
        period2 = 106
    elif year == 2015:
        period1 = 106
        period2 = 158
    elif year == 2016:
        period1 = 158
        period2 = 210
    elif year == 2017:
        period1 = 210
        period2 = 262
    elif year == 2018:
        period1 = 262
        period2 = 314
    elif year == 2019:
         period1 = 314
         period2 = 367
    elif year == 2020:
        period1 = 367
        period2 = 419
    elif year == 2021:
        period1 = 419
        period2 = 432
    else:
        print('error')

    rd = pd.read_excel('esgdata.xlsx',
                        sheet_name='return', header=0)
    capex13 = pd.read_excel('esgdata.xlsx',
                        sheet_name='capex', header=0)

        # 13
    capex13 = capex13.loc[capex13['Assessment Year'] == year]
    capex13 = capex13.sort_values(['Market Value - Total - Fiscal'], ascending=[True])

    length13 = round(0.1 * len(capex13))
    b13 = (len(capex13) - length13)
    datas = []
    datab = []
    for row in range(length13):
        datas.append(capex13.iloc[row])
        datab.append(capex13.iloc[b13 + row])

    caps = pd.DataFrame(datas)
    capb = pd.DataFrame(datab)

    caps = caps.sort_values(['Capital Expenditures'], ascending=[True])
    capb = capb.sort_values(['Capital Expenditures'], ascending=[True])

    length13 = round(0.3 * len(caps))
    b13 = (len(caps) - length13)
    datasg = []
    datasv = []
    databg = []
    databv = []

    for row in range(length13):
        datasg.append(caps.iloc[row])
        datasv.append(caps.iloc[b13 + row])
        databg.append(capb.iloc[row])
        databv.append(capb.iloc[b13 + row])

    capsg = pd.DataFrame(datasg)
    capsv = pd.DataFrame(datasv)
    capbg = pd.DataFrame(databg)
    capbv = pd.DataFrame(databv)

    length13 = round(len(capsg))
    b13 = (len(capsg) - length13)
    for i in range(period1, period2):
        c = 0
        d = 0
        e = 0
        f = 0
        for j in range(length13):
            c = c + (rd.iloc[i, rd.columns.get_loc(capsv.iloc[j, capsv.columns.get_loc('CUSIP')])])
            e = e + (rd.iloc[i, rd.columns.get_loc(capsg.iloc[(b13 + j), capsg.columns.get_loc('CUSIP')])])

            d = d + (rd.iloc[i, rd.columns.get_loc(capbv.iloc[j, capbv.columns.get_loc('CUSIP')])])
            f = f + (rd.iloc[i, rd.columns.get_loc(capbg.iloc[(b13 + j), capbg.columns.get_loc('CUSIP')])])
        a = ((1 / 2) * (1 / length13) * (c - e)) + ((1 / 2) * (1 / length13) * (d - f))
        capex.append(a)
    return capex

def imnv(year,imn):
    if year == 2013:
        period1 = 1
        period2 = 53
    elif year == 2014:
        period1 = 53
        period2 = 106
    elif year == 2015:
        period1 = 106
        period2 = 158
    elif year == 2016:
        period1 = 158
        period2 = 210
    elif year == 2017:
        period1 = 210
        period2 = 262
    elif year == 2018:
        period1 = 262
        period2 = 314
    elif year == 2019:
        period1 = 314
        period2 = 367
    elif year == 2020:
        period1 = 367
        period2 = 419
    elif year == 2021:
        period1 = 419
        period2 = 432
    else:
        print('error')

    rd = pd.read_excel('esgdata.xlsx',
                           sheet_name='return', header=0)
    es= pd.read_excel('esgdata.xlsx',
                           sheet_name='imn', header=0)


    # 13
    esg13 = es.loc[es['Assessment Year'] == year]
    esg13 = esg13.sort_values(['Market Value - Total - Fiscal'], ascending=[True])

    length13 = round(0.5*len(esg13))
    b13 = (len(esg13)- length13)
    datas = []
    datab = []
    for row in range(length13):
        datas.append(esg13.iloc[row])
        datab.append(esg13.iloc[b13+row])

    esgs = pd.DataFrame(datas)
    esgb = pd.DataFrame(datab)

    esgs = esgs.sort_values(['capex'], ascending=[True])
    esgb = esgb.sort_values(['capex'], ascending=[True])

    length13 = round(0.3*len(esgs))
    b13 = (len(esgs)- length13)
    datash = []
    datasl = []
    databh = []
    databl = []

    for row in range(length13):
        datasl.append(esgs.iloc[row])
        datash.append(esgs.iloc[b13+row])
        databl.append(esgb.iloc[row])
        databh.append(esgb.iloc[b13+row])

    esgsl = pd.DataFrame(datasl)
    esgsh = pd.DataFrame(datash)
    esgbl = pd.DataFrame(databl)
    esgbh = pd.DataFrame(databh)

    esgsl = esgsl.sort_values(['Score Value env'], ascending=[True])
    esgbl = esgbl.sort_values(['Score Value env'], ascending=[True])
    esgsh = esgsh.sort_values(['Score Value env'], ascending=[True])
    esgbh = esgbh.sort_values(['Score Value env'], ascending=[True])

    length13 = round(0.3*len(esgsh))
    b13 = (len(esgsh)- length13)
    datashg = []
    datashv = []
    dataslg = []
    dataslv = []
    databhg = []
    databhv = []
    datablg = []
    datablv = []


    for row in range(length13):
        dataslg.append(esgsl.iloc[row])
        datablg.append(esgbl.iloc[row])
        datashg.append(esgsh.iloc[row])
        databhg.append(esgbh.iloc[row])
        dataslv.append(esgsl.iloc[b13+row])
        datablv.append(esgbl.iloc[b13+row])
        datashv.append(esgsh.iloc[b13+row])
        databhv.append(esgbh.iloc[b13+row])
    esgslg = pd.DataFrame(dataslg)
    esgshg = pd.DataFrame(datashg)
    esgslv = pd.DataFrame(dataslv)
    esgshv = pd.DataFrame(datashv)
    esgblg = pd.DataFrame(datablg)
    esgbhg = pd.DataFrame(databhg)
    esgblv = pd.DataFrame(datablv)
    esgbhv =pd.DataFrame(databhv)
    length13 = round(len(esgshv))
    b13 = (len(esgshv)- length13)
    for i in range (period1,period2):
        c = 0
        d = 0
        e = 0
        f = 0
        a  = 0
        for j in range(length13):
            c = c+ (rd.iloc[i, rd.columns.get_loc(esgshv.iloc[j, esgshv.columns.get_loc('CUSIP')])])
            e = e +(rd.iloc[i, rd.columns.get_loc(esgslg.iloc[(b13+j), esgslg.columns.get_loc('CUSIP')])])

            d= d +(rd.iloc[i, rd.columns.get_loc(esgbhv.iloc[j, esgbhv.columns.get_loc('CUSIP')])])
            f = f+(rd.iloc[i, rd.columns.get_loc(esgblg.iloc[(b13 + j), esgblg.columns.get_loc('CUSIP')])])
        a = ((1/2)*(1/length13)*(c-e))+((1/2)*(1/length13)*(d-f))
        imn.append(a)

    return imn
def imnv2(year,imn2):
    if year == 2013:
        period1 = 1
        period2 = 53
    elif year == 2014:
        period1 = 53
        period2 = 106
    elif year == 2015:
        period1 = 106
        period2 = 158
    elif year == 2016:
        period1 = 158
        period2 = 210
    elif year == 2017:
        period1 = 210
        period2 = 262
    elif year == 2018:
        period1 = 262
        period2 = 314
    elif year == 2019:
        period1 = 314
        period2 = 367
    elif year == 2020:
        period1 = 367
        period2 = 419
    elif year == 2021:
        period1 = 419
        period2 = 432
    else:
        print('error')

    rd = pd.read_excel('esgdata.xlsx',
                           sheet_name='return', header=0)
    es= pd.read_excel('esgdata.xlsx',
                           sheet_name='imn', header=0)


    # 13
    esg13 = es.loc[es['Assessment Year'] == year]
    esg13 = esg13.sort_values(['Market Value - Total - Fiscal'], ascending=[True])

    length13 = round(0.5*len(esg13))
    b13 = (len(esg13)- length13)
    datas = []
    datab = []
    for row in range(length13):
        datas.append(esg13.iloc[row])
        datab.append(esg13.iloc[b13+row])

    esgs = pd.DataFrame(datas)
    esgb = pd.DataFrame(datab)

    esgs = esgs.sort_values(['Score Value env'], ascending=[True])
    esgb = esgb.sort_values(['Score Value env'], ascending=[True])

    length13 = round(0.3*len(esgs))
    b13 = (len(esgs)- length13)
    datash = []
    datasl = []
    databh = []
    databl = []

    for row in range(length13):
        datasl.append(esgs.iloc[row])
        datash.append(esgs.iloc[b13+row])
        databl.append(esgb.iloc[row])
        databh.append(esgb.iloc[b13+row])

    esgsl = pd.DataFrame(datasl)
    esgsh = pd.DataFrame(datash)
    esgbl = pd.DataFrame(databl)
    esgbh = pd.DataFrame(databh)

    esgsl = esgsl.sort_values(['capex'], ascending=[True])
    esgbl = esgbl.sort_values(['capex'], ascending=[True])
    esgsh = esgsh.sort_values(['capex'], ascending=[True])
    esgbh = esgbh.sort_values(['capex'], ascending=[True])
    length13 = round(0.3*len(esgsh))
    b13 = (len(esgsh)- length13)
    datashg = []
    datashv = []
    dataslg = []
    dataslv = []
    databhg = []
    databhv = []
    datablg = []
    datablv = []


    for row in range(length13):
        dataslg.append(esgsl.iloc[row])
        datablg.append(esgbl.iloc[row])
        datashg.append(esgsh.iloc[row])
        databhg.append(esgbh.iloc[row])
        dataslv.append(esgsl.iloc[b13+row])
        datablv.append(esgbl.iloc[b13+row])
        datashv.append(esgsh.iloc[b13+row])
        databhv.append(esgbh.iloc[b13+row])
    esgslg = pd.DataFrame(dataslg)
    esgshg = pd.DataFrame(datashg)
    esgslv = pd.DataFrame(dataslv)
    esgshv = pd.DataFrame(datashv)
    esgblg = pd.DataFrame(datablg)
    esgbhg = pd.DataFrame(databhg)
    esgblv = pd.DataFrame(datablv)
    esgbhv =pd.DataFrame(databhv)

    length13 = round(len(esgshv))
    b13 = (len(esgshv)- length13)

    for i in range (period1,period2):
        c = 0
        d = 0
        e = 0
        f = 0
        a = 0
        for j in range(length13):
            c = c+ (rd.iloc[i, rd.columns.get_loc(esgshv.iloc[j, esgshv.columns.get_loc('CUSIP')])])
            e = e +(rd.iloc[i, rd.columns.get_loc(esgslg.iloc[(b13+j), esgslg.columns.get_loc('CUSIP')])])

            d= d +(rd.iloc[i, rd.columns.get_loc(esgbhv.iloc[j, esgbhv.columns.get_loc('CUSIP')])])
            f = f+(rd.iloc[i, rd.columns.get_loc(esgblg.iloc[(b13 + j), esgblg.columns.get_loc('CUSIP')])])
        a = ((1/2)*(1/length13)*(c-e))+((1/2)*(1/length13)*(d-f))
        imn2.append(a)

    return imn2




def main():
    vr = pd.read_excel('esgdata.xlsx',
                          sheet_name='variables', header=0)
    x1 = ['rmf','smb','hml','vol','mom'],['rmf','smb','hml','vol','mom','rd'],['rmf','smb','hml','vol','mom','env'],['rmf', 'smb', 'hml', 'vol', 'mom', 'capex'],['rmf','smb','hml','vol','mom','pmb'],['rmf','smb','hml','vol','mom','pmb2'],['rmf', 'smb', 'hml', 'vol', 'mom', 'imn'],['rmf', 'smb', 'hml', 'vol', 'mom', 'imn2']

    constant = []
    rmfi = []
    smbi = []
    hmli =[]
    voli =[]
    momi = []
    rdi =['&']
    envi =['&','&']
    capexi =['&','&','&']
    pmbi =['&','&','&','&']
    pmb2i =['&','&','&','&','&']
    imni =['&','&','&','&','&','&']
    imn2i = ['&','&','&','&','&','&','&']
    for values in x1:
        regress(values,constant,rmfi,smbi,hmli,voli,momi,rdi,envi,capexi,pmbi,pmb2i,imni,imn2i)

    df = pd.DataFrame([list(constant), list(rmfi),list(smbi), list(hmli),list(voli), list(momi),list(rdi), list(envi), list(capexi),list(pmbi), list(pmb2i),list(imni), list(imn2i)], columns=["Model1", "Model2", "Model3", "Model4","Model5", "Model6", "Model7", "Model8"])
    df.index = ['Constant', 'RMF', 'SMB', 'HML','VOL', 'MOM', 'RD', 'ENV','CAPEX','PMB', 'PMB2', 'IMN', 'IMN2']
    pd.DataFrame(df).style.to_latex()
    smb = []
    hml = []
    esg = []
    esg2 = []
    esg3 = []
    rd2 = []
    capex = []
    imn = []
    imn2 = []
    momentum = []
    volatillity = []

    # for i in range(2013,2022):
    #     # smbv(i,smb)
    #     # hmlv(i,hml)
    #     esgv(i,esg)
    #     esgv2(i, esg2)
    #     esgv3(i, esg3)
    #     rdv2(i, rd2)
    #     capexv(i, capex)
    #     imnv(i,imn)
    #     imnv2(i,imn2)
    # print("smb")
    # for values in smb:
    # #     print(values)
    # print("esg1")
    # for values in esg:
    #     print(values)
    # print("esg2")
    # for values in esg2:
    #     print(values)
    # for values in rd2:
    #     print(values)
    # print("esg3")
    # for values in esg3:
    #     print(values)
    # print("hmml")
    # for values in hml:
    #     print(values)
    # for values in capex:
    #     print(values)
    # for values in imn:
    #     print(values)
    # print("imn2")
    # for values in imn2:
    #     print(values)
    # vol(volatillity)
    # for i in range(1,101):
    #     print(i)
    # #     mom(i,momentum)
    #     vol(i,volatillity)
    # # for values in momentum:
    # #     print(values)
    # for values in volatillity:
    #     print(values)

    #
    # table = vr.describe()
    # cor = vr.corr()
    # cov = vr.cov()
    #
    #
    # sn.heatmap(table, annot=True, fmt='g')
    # plt.show()
    #
    # sn.heatmap(cor, annot=True, fmt='g')
    # plt.show()
    #
    # sn.heatmap(cov, annot=True, fmt='g')
    # plt.show()







if __name__ == "__main__":
    main()