import pandas as pd
import numpy as np


class Data:
    def __init__(self, root_dir='../docs/Python_codes/'):
        self.root_dir = root_dir
        self.data = None

        # data location
        self.loc1 = self.root_dir + "data.csv"
        self.loc2 = self.root_dir + "demogr.csv"
        self.loc3 = self.root_dir + "demogr_means.csv"
        self.loc4 = self.root_dir + "demogr_year_region_id.csv"
        self.loc5 = self.root_dir + "demogr_iqr.csv"
        self.loc6 = self.root_dir + "demogr_std.csv"
        self.r = self.root_dir + "V.csv"

        self.data = None
        self.x1 = None
        self.x2 = None
        self.s_jt = None
        self.IV = None
        self.v = None
        self.cdindex = None
        self.cdid = None
        self.cdid_demogr = None
        self.demogr = None
        self.demogr_means = None
        self.demogr_std = None
        self.demogr_iqr = None
        self.id = None
        self.id_demo = None
        self.K1 = None
        self.K2 = None
        self.nmkt = None
        self.ns = None

    def load_data(self):
        # load data
        data = pd.read_csv(self.loc1, sep=',', names=list(range(1, 189)), encoding='cp1252').astype(float)
        demogr = pd.read_csv(self.loc2, sep=',', names=list(range(1, 1001)), encoding='cp1252').astype(float)
        demogr_means = pd.read_csv(self.loc3, sep=',', names=list(range(1, 1001)), encoding='cp1252').astype(float)
        demogr_year_region_id = pd.read_csv(self.loc4, sep=',', names=list(range(1, 3)), encoding='cp1252').astype(float)
        demogr_iqr = pd.read_csv(self.loc5, sep=',', names=list(range(1, 1001)), encoding='cp1252').astype(float)
        demogr_std = pd.read_csv(self.loc6, sep=',', names=list(range(1, 1001)), encoding='cp1252').astype(float)

        # column names
        names = ['id_colid_market_col', 'ms_col', 'dummy_firm_start', 'dummy_firm_end', 'dummy_year_start',
                 'dummy_year_end', 'dummy_region_start', 'dummy_region_end', 'dummy_firmname_start',
                 'dummy_firmname_end',
                 'dummy_macroregion_start', 'dummy_macroregion_end', 'price_col', 'deductible_col', 'n_medications_col',
                 'drug_frf_col', 'd_cov_gap_col', 'd_enhance_col', 'top_drug_col', 'in_area_flag_col', 'vintage_col',
                 'tier_12_col', 'd_below_lagged_col', 'ms_lis_firm_col', 'ms_lis_firm_lagged_col', 'id_col',
                 'id_market_col', 'ms_col', 'dummy_firm_start', 'dummy_firm_end', 'dummy_year_start', 'dummy_year_end',
                 'dummy_region_start', 'dummy_firmname_start', 'dummy_firmname_end', 'dummy_macroregion_start',
                 'dummy_macroregion_end', 'deductible_col', 'n_medications_col', 'drug_frf_col', 'd_cov_gap_col',
                 'd_enhance_col', 'op_drug_col', 'in_area_flag_col', 'vintage_col', 'tier_12_col', 'd_below_lagged_col',
                 'ms_lis_firm_col', 'ms_lis_firm_lagged_col', 'iv_start_col', 'iv_end_col', 'bid_col', 'gamma_col',
                 'true_weight_col', 'nat_weight_col', 'groupA_col', 'group_region_col', 'group_bid_test_col',
                 'ms_lischoosers_col', 'tot_lis_col', 'tot_lis_choosers_col', 'market_sizes_col', 'ms_lis_col',
                 'indicator_bench_col', 'contract_id_col', 'benefit_type_col', 'plan_name_col', 'lis_col']
        x1_names = ['id_colid_market_col', 'ms_col', 'dummy_firm_start', 'dummy_firm_end', 'dummy_year_start',
                    'dummy_year_end', 'dummy_region_start', 'dummy_region_end', 'dummy_firmname_start',
                    'dummy_firmname_end', 'dummy_macroregion_start', 'dummy_macroregion_end', 'deductible_col',
                    'n_medications_col', 'drug_frf_col', 'd_cov_gap_col', 'd_enhance_col', 'top_drug_col',
                    'in_area_flag_col', 'vintage_col', 'tier_12_col', 'd_below_lagged_col', 'ms_lis_firm_col',
                    'ms_lis_firm_lagged_col', 'id_col', 'id_market_col', 'ms_col', 'dummy_firm_start', 'dummy_firm_end',
                    'dummy_year_start', 'dummy_year_end', 'dummy_region_start', 'dummy_firmname_start',
                    'dummy_firmname_end', 'dummy_macroregion_start', 'dummy_macroregion_end', 'deductible_col',
                    'n_medications_col', 'drug_frf_col', 'd_cov_gap_col', 'd_enhance_col', 'op_drug_col',
                    'in_area_flag_col', 'vintage_col', 'tier_12_col', 'd_below_lagged_col', 'ms_lis_firm_col',
                    'ms_lis_firm_lagged_col']
        x2_names = ['price_col']
        D_names = list(range(1, 1001))
        IV_names = ['iv_start_col', 'iv_end_col', 'bid_col', 'gamma_col', 'true_weight_col', 'nat_weight_col',
                    'groupA_col',
                    'group_region_col', 'group_bid_test_col', 'ms_lischoosers_col', 'tot_lis_col',
                    'tot_lis_choosers_col',
                    'market_sizes_col', 'ms_lis_col', 'indicator_bench_col', 'contract_id_col', 'benefit_type_col',
                    'plan_name_col', 'lis_col']

        id_col = 1  # id market plan
        id_market_col = 2  # id market
        ms_col = 3  # market shares
        dummy_firm_start = 4  # first column of firm id's dummies
        dummy_firm_end = 79  # last column of firm id's dummies
        dummy_year_start = 80  # first column of year's dummies
        dummy_year_end = 85  # last column of year's dummies
        dummy_region_start = 86  # first column of region's dummies
        dummy_region_end = 119  # last column of region's dummies
        dummy_firmname_start = 120  # first column of firm name
        dummy_firmname_end = 140  # last column of firm name
        dummy_macroregion_start = 141  # first column of firm name
        dummy_macroregion_end = 143  # last column of firm name
        price_col = 144
        deductible_col = price_col + 1
        n_medications_col = price_col + 2  # col 146 - try adding this to x1
        drug_frf_col = price_col + 3
        d_cov_gap_col = price_col + 4
        d_enhance_col = price_col + 5
        top_drug_col = price_col + 6
        in_area_flag_col = price_col + 7
        vintage_col = price_col + 8
        tier_12_col = price_col + 9
        d_below_lagged_col = price_col + 10
        ms_lis_firm_col = price_col + 11
        ms_lis_firm_lagged_col = price_col + 12

        # LIST OF IV'S
        iv_start_col = price_col + 13
        iv_end_col = iv_start_col + 12
        bid_col = iv_start_col + 15
        gamma_col = iv_start_col + 16
        true_weight_col = iv_start_col + 17
        nat_weight_col = iv_start_col + 18
        groupA_col = iv_start_col + 19
        group_region_col = iv_start_col + 20
        group_bid_test_col = iv_start_col + 21
        ms_lischoosers_col = iv_start_col + 22
        tot_lis_col = iv_start_col + 233
        tot_lis_choosers_col = iv_start_col + 24
        market_sizes_col = iv_start_col + 25
        ms_lis_col = iv_start_col + 26
        indicator_bench_col = iv_start_col + 27  # =1 if below benchmark plan
        contract_id_col = iv_start_col + 28
        benefit_type_col = iv_start_col + 29
        plan_name_col = iv_start_col + 30
        lis_col = iv_start_col + 31

        # #cleaning
        # #drop missing iv, dates, no-pharmacy plans

        # missing IV
        data.loc[:, iv_start_col + 3] = data.loc[:, iv_start_col + 3].fillna(value=0)
        data.loc[:, iv_start_col + 4] = data.loc[:, iv_start_col + 4].fillna(value=0)
        data.loc[:, iv_start_col + 5] = data.loc[:, iv_start_col + 5].fillna(value=0)
        data.loc[:, iv_start_col + 6] = data.loc[:, iv_start_col + 6].fillna(value=0)
        data.loc[:, iv_start_col + 11] = data.loc[:, iv_start_col + 11].fillna(value=0)
        # drop dates
        index = (data.loc[:, 1] < 11000000) & (data.loc[:, 1] > 7000000)
        data = data.loc[index, :]
        # drop plans with no pharmacy? check column 151 is this one
        index = (data.loc[:, in_area_flag_col] != 0) & (data.loc[:, in_area_flag_col] < 10000)
        data = data.loc[index, :]

        # re-scaling
        scale = 100
        ns = 500
        data.loc[:,
        [price_col, deductible_col, n_medications_col, drug_frf_col, d_cov_gap_col, d_enhance_col, top_drug_col,
         vintage_col, tier_12_col, dummy_macroregion_start, 142, dummy_macroregion_end]] = data.loc[:,
                                                                                           [price_col,
                                                                                            deductible_col,
                                                                                            n_medications_col,
                                                                                            drug_frf_col,
                                                                                            d_cov_gap_col,
                                                                                            d_enhance_col,
                                                                                            top_drug_col,
                                                                                            vintage_col,
                                                                                            tier_12_col,
                                                                                            dummy_macroregion_start,
                                                                                            142,
                                                                                            dummy_macroregion_end]].div(scale)
        data.loc[:, dummy_year_start:dummy_year_end] = data.loc[:, dummy_year_start:dummy_year_end].div(scale)
        data.loc[:, dummy_region_start:dummy_region_end] = data.loc[:, dummy_region_start:dummy_region_end].div(scale)
        data.loc[:, dummy_firmname_start:dummy_firmname_end] = data.loc[:, dummy_firmname_start:dummy_firmname_end].div(scale)
        data.loc[:, in_area_flag_col] = data.loc[:, in_area_flag_col].div(1000)

        demogr.loc[:, 1:500] = demogr.loc[:, 1:500].astype(float).div(1000)
        demogr_means.loc[:, 1:ns] = demogr_means.loc[:, 1:ns].astype(float).div(1000)

        # IV - matrix

        IV = data[
            list(range(iv_start_col, iv_end_col + 1)) + [deductible_col, d_cov_gap_col, d_enhance_col, tier_12_col,
                                                         top_drug_col, in_area_flag_col, vintage_col] + list(
                range(dummy_macroregion_start, dummy_macroregion_end)) + list(
                range(dummy_year_start + 1, dummy_year_end - 2)) + [dummy_firmname_start, dummy_firmname_start + 10,
                                                                    dummy_firmname_start + 13,
                                                                    dummy_firmname_start + 16,
                                                                    dummy_firmname_end]]
        # these columns give non singuality: 6-10
        IV = IV.iloc[:, 0:6].join(IV.iloc[:, 11:29])

        # Cleaning missing IV
        index = (np.sum(IV.T.isna(), axis=0) == 0)
        data = data.loc[index, :]
        IV = IV.loc[index, :]

        # reindexing IV
        IV.index = pd.RangeIndex(len(IV.index))
        IV.columns = pd.RangeIndex(len(IV.columns))

        # x1, x2
        constant = pd.Series(np.ones(np.shape(data[1])), index=data.index, name='constant')  # add constant
        x11 = data[[price_col, deductible_col, drug_frf_col, d_cov_gap_col, d_enhance_col, tier_12_col, top_drug_col,
                    in_area_flag_col, vintage_col] + list(range(dummy_macroregion_start, dummy_macroregion_end)) + list(
            range(dummy_year_start + 1, dummy_year_end - 2)) + [dummy_firmname_start, dummy_firmname_start + 10,
                                                                dummy_firmname_start + 13, dummy_firmname_start + 15,
                                                                dummy_firmname_start + 16, dummy_firmname_end]]
        # try to add n_medications_col
        # x11=data[[price_col, deductible_col, drug_frf_col, d_cov_gap_col, d_enhance_col, tier_12_col, top_drug_col, in_area_flag_col, vintage_col,n_medications_col]+list(range(dummy_macroregion_start,dummy_macroregion_end))+list(range(dummy_year_start+1,dummy_year_end-2))+[dummy_firmname_start,dummy_firmname_start+10,dummy_firmname_start+13,dummy_firmname_start+15,dummy_firmname_start+16,dummy_firmname_end]]

        x1 = x11.join(constant)
        col = x1.columns.tolist()
        col = col[-1:] + col[:-1]
        x1 = x1[col]

        x2 = data.loc[:, price_col].values.reshape(len(x1), 1)
        # x2 = data.loc[:,[price_col, deductible_col]].values.reshape(len(x1),2) #try to add another variable to x2

        # Cleaning x1:eliminate dummies that are always 0 (companies that never offer a plan in 2006)
        index = np.sum(x1, axis=0)
        x1 = x1.loc[:, index != 0]

        # reindexing
        x1.index = pd.RangeIndex(len(x1.index))  # reindex x1
        x1.columns = pd.RangeIndex(len(x1.columns))

        # indices
        id = data.loc[:, id_col]  # id of each observation
        id_demo = data.loc[:, id_market_col].unique()  # number of markets

        # The  vector  below relates each observation to the market it is in / both for data and demographics

        cdid = list()
        cdindex = [-1]
        cdid_demogr = list()

        for i in range(len(id_demo)):
            nbrand_market = np.sum(data.loc[:, id_market_col] == id_demo[i])
            cdid.extend(i * np.ones(nbrand_market))
            cdindex.append(cdindex[-1] + nbrand_market)
            # create the cdid for the demographics
            index_market = data.loc[data[id_market_col] == id_demo[i]].index
            i_index_region = data.loc[index_market[0], dummy_region_start:dummy_region_end] == (1 / scale)
            index_region = np.where(i_index_region)[0] + 1  # +1 since id_region [1,34]
            i_index_year = data.loc[index_market[0], dummy_year_start:dummy_year_end] == (1 / scale)
            index_year = np.where(i_index_year)[0] + 2005
            index_demogr = demogr_year_region_id[
                (demogr_year_region_id[1] == index_year[0]) &
                (demogr_year_region_id[2] == index_region[0])
                ].index
            if np.isnan(index_demogr.values) == 1:
                print(['Demographics not found for region ' + np.array2string(
                    index_region) + ' and year ' + np.array2string(index_year)])
            else:
                cdid_demogr.extend(index_demogr.values * np.ones(nbrand_market, int))  ## [0,135]

        cdindex = cdindex[1:137]
        cdindex = np.asarray(cdindex, dtype=int)

        # create instruments based on demographics
        mean_income = demogr_means.loc[cdid_demogr, 1]
        mean_diffall = demogr_means.loc[cdid_demogr, ns + 1]
        std_income = demogr_std.loc[cdid_demogr, 1]
        iqr_income = demogr_iqr.loc[cdid_demogr, 1]
        iqr_diffall = demogr_iqr.loc[cdid_demogr, ns + 1]  #

        IV1 = pd.DataFrame(np.concatenate(
            [np.array([IV.loc[:, 13]]) * np.array([mean_income]), np.array([IV.loc[:, 13]]) * np.array([mean_diffall]),
             np.array([IV.loc[:, 13]]) * np.array([iqr_income]), np.array([IV.loc[:, 13]]) * np.array([iqr_diffall])],
            axis=0).T, index=IV.index, columns=[24, 25, 26, 27])
        IV = IV.merge(IV1, right_index=True, left_index=True)  # Try to run without these IV

        # other variables
        K1 = x1.shape[1]  # number of product characteristics
        K2 = x2.shape[1]  #
        s_jt = data.loc[:, ms_col]
        s_jt = np.reshape(s_jt.values, (np.shape(s_jt)[0], 1))  # observed market share for product j at time t
        nmkt = len(id_demo)
        v = pd.read_csv(self.r, sep=',', names=list(range(1000)))

        # v = pd.DataFrame(np.random.randn(nmkt,demogr.shape[1]))

        self.data = data
        self.x1 = x1
        self.x2 = x2
        self.s_jt = s_jt
        self.IV = IV
        self.v = v
        self.cdindex = cdindex
        self.cdid = cdid
        self.cdid_demogr = cdid_demogr
        self.demogr = demogr
        self.demogr_means = demogr_means
        self.demogr_std = demogr_std
        self.demogr_iqr = demogr_iqr
        self.id = id
        self.id_demo = id_demo
        self.K1 = K1
        self.K2 = K2
        self.nmkt = nmkt
        self.ns = ns

    def summary(self):
        data = self.data
        print('the number of characteristics in X1 is: ' + str(data.x1.shape[1]))
        print('the number of characteristics in X2 is: ' + str(data.x2.shape[1]))
        print('Object cdid dimensions: ' + str(np.shape(data.cdid)))
        print('The dimensions of object cdindex are      ' + str(np.shape(data.cdindex)))
        print('The number of instruments for the price is ' + str(np.shape(data.IV)[1]))
        print('The dimensions of object IV are ' + str(np.shape(data.IV)))
        print('size id_demo: ' + str(np.shape(data.id_demo)))
        print("size of cdid is " + str(np.shape((data.cdid))))
        print("size of cdindex is " + str(np.shape((data.cdindex))))
        print("size of cdid_demogr is " + str(np.shape((data.cdid_demogr))))


if __name__ == "__main__":
    data_instance = Data()
    data_instance.load_data()
    data_instance.summary()
