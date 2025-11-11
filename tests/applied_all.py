from citest.data import Dataset
from citest.imputer import MidasImputer
from citest.classifier import RandomForest
from citest import CIMissTest

import pandas as pd
import numpy as np
import random

# Set seed
import torch
np.random.seed(42)
torch.manual_seed(42)

##########################
## Clark and Dolan 2021 ##
##########################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/AJPS/Clark and Dolan 2021/cd2021_full.csv"
)

#############
## Model 1 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="count_pa", expl_vars=["absidealimportantdiff", "board", "colony", "unsc", "USaid", "CHaid", "gdppc", "dservtoGDP", "dshorttoexports", "inflation", "debttoGDP", "FDItoGDP", "polity2", "openness", "war", "elec", "IMF", "crisis", "ccode"]
)


# Define the test object
pol_test = CIMissTest(
    pol_dataset,
#    imputer=MidasImputer,
#    classifier=RandomForest,
#    n_folds=10,
#    m=10,
#    classifier_args={"n_estimators": 20, "n_jobs": 8},
#    imputer_args={"hidden_layers": [8, 4, 2], "epochs": 500},
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 2 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="numcats", expl_vars=["absidealimportantdiff", "board", "colony", "unsc", "USaid", "CHaid", "gdppc", "dservtoGDP", "dshorttoexports", "inflation", "debttoGDP", "FDItoGDP", "polity2", "openness", "war", "elec", "IMF", "crisis", "ccode"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

###############
## Good 2024 ##
###############

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/APSR/Good 2024/PADD_Agreement Level.csv"
)

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="GeWom", expl_vars=["FemDel_P", "GDI", "WomParl", "SEP_Fem", "TeenPreg", "NYT_p", "UNSCR", "Press_UNSC", "ImUN", "ImOth", "NAP", "PolInt_Cmb", "JobEql_Cmb", "LeadPol_Cmb", "state_prev_avg", "female_combatants_exs"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

###################
## Claassen 2020 ##
###################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/APSR/Claassen 2020/c2020_full.csv"
)

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="SupDem_trim_dl1", expl_vars=["SupDem_trim_l1", "SupDem_trim_l2", "Libdem_z_l1", "Libdem_z_d", "lnGDP_imp_d", "Corrup_TI_z_d", "Corrup_TI_z_l1"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#########################
## Dellmuth et al 2022 ##
#########################

# Load in the dataset using pandas
pol_data = pd.read_stata(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/APSR/Dellmuth et al 2022/Auxiliary datasets/dyads_recodedAPSR.dta"
)

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="DIFFconfios", expl_vars=["DIFFedu", "DIFFfinsathh", "DIFFlr", "DIFFgal", "DIFFfeelworld", "DIFFfeelcountry", "DIFFconfgov", "DIFFsatis", "DIFFage", "DIFFsex", "country"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

###########################
## Gamm and Kousser 2021 ##
###########################

# Load in the dataset using pandas
pol_data = pd.read_stata(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/APSR/Gamm and Kousser 2021/PoliticsProsperityJune2021.dta"
)

#############
## Model 2 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="Education_pc", expl_vars=["leg_party_competition", "Statewide_Competition", "house_dem",  "senate_dem", "gov_dem", "CPI_per_capita_income", "foreignborn_pct", "black_pct", "othernonwhite_pct", "urban_pct", "year_1890", "year_1900", "year_1910", "year_1930", "year_1940", "year_1960", "year_1970", "year_1980", "state"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 4 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="HealthSewerSanitation_pc", expl_vars=["leg_party_competition", "Statewide_Competition", "house_dem",  "senate_dem", "gov_dem", "CPI_per_capita_income", "foreignborn_pct", "black_pct", "othernonwhite_pct", "urban_pct", "year_1890", "year_1900", "year_1910", "year_1930", "year_1940", "year_1960", "year_1970", "year_1980", "state"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 6 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="Transportation_pc", expl_vars=["leg_party_competition", "Statewide_Competition", "house_dem", "senate_dem", "gov_dem", "CPI_per_capita_income", "foreignborn_pct", "black_pct", "othernonwhite_pct", "urban_pct", "year_1890", "year_1900", "year_1910", "year_1930", "year_1940", "year_1960", "year_1970", "year_1980", "state"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

######################
## Nyrup et al 2024 ##
######################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/APSR/Nyrup et al 2024/1_data/df_consolidatingprogress_V1.csv"
)

#############
## Model 3 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="share_female", expl_vars=["lag_democracy_stock_poly_95", "lag_gdp_cap_pwt_ln", "lag_wb_oilrev", "lag_growth_pwt", "lag_wdi_popurb", "lag_pop_pwt_ln", "year", "country_isocode"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 4 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="share_female", expl_vars=["lag_democracy_stock_poly_95", "lag_e_pelifeex", "lag_wb_infantmortality", "lag_wb_primaryschoolenrolment", "year", "country_isocode"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 5 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="share_female", expl_vars=["lag_democracy_stock_poly_95", "lag_v2x_gender", "lag_v2lgfemleg", "lag_ciri_wopol", "lag_ciri_wecon", "lag_female_leader", "year", "country_isocode"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 6 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="share_female", expl_vars=["lag_democracy_stock_poly_95", "lag_v2xcl_rol", "lag_v2xcl_prpty", "lag_v2x_rule", "lag_v2x_jucon", "lag_v2xlg_legcon", "lag_v2x_corr", "lag_v2clstown", "lag_v2xcs_ccsi", "lag_v2xps_party", "year", "country_isocode"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#################
## Hansen 2022 ##
#################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/APSR/Hansen 2022/h2022_full.csv"
)

#############
## Model 2 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="d_logunemp_ilo", expl_vars=["ld_logunemp_ilo", "d_KOFEcGIdf", "l_KOFEcGIdf", "d_fiscal", "l_loggdppc", "leftgov", "l_eldem", "l_EquityBankRatio", "l_logtotresgdp", "l_default", "l_logcpi", "d_range", "l_range", "peg", "CrBubble5dum", "l_logcpi_Reg", "l_logevol__Reg", "uscrisis", "t_lastBCsys"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 4 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="d_logdomcredit", expl_vars=["ld_logdomcredit", "d_KOFEcGIdf", "l_KOFEcGIdf", "d_fiscal", "l_loggdppc", "leftgov", "l_eldem", "l_EquityBankRatio", "l_logtotresgdp", "l_default", "l_logcpi", "d_range", "l_range", "peg", "CrBubble5dum", "l_logcpi_Reg", "l_logevol__Reg", "uscrisis", "t_lastBCsys"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 6 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="d_logstockvalue", expl_vars=["ld_logstockvalue", "d_KOFEcGIdf", "l_KOFEcGIdf", "d_fiscal", "l_loggdppc", "leftgov", "l_eldem", "l_EquityBankRatio", "l_logtotresgdp", "l_default", "l_logcpi", "d_range", "l_range", "peg", "CrBubble5dum", "l_logcpi_Reg", "l_logevol__Reg", "uscrisis", "t_lastBCsys"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results


#########################
## Aklin and Kern 2021 ##
#########################

# Load in the dataset using pandas
pol_data = pd.read_stata(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/AJPS/Aklin and Kern 2021/AklinKernReplicationData.dta"
)

#############
## Model 3 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="finreform", expl_vars=["lvaw", "year", "loggdp", "loggdpcapita", "polity2", "pegtype", "ccode"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 6 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="entrybarriers", expl_vars=["lvaw", "year", "loggdp", "loggdpcapita", "polity2", "pegtype", "ccode"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 9 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="privatization", expl_vars=["lvaw", "year", "loggdp", "loggdpcapita", "polity2", "pegtype", "ccode"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 12 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="deposit_insurance", expl_vars=["lvaw", "year", "loggdp", "loggdpcapita", "polity2", "pegtype", "ccode"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

##############
## Model 15 ##
##############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="ckaopen2010", expl_vars=["lvaw", "year", "loggdp", "loggdpcapita", "polity2", "pegtype", "ccode"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

####################
## Leipziger 2022 ##
####################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/AJPS/Leipziger 2024/l2024_full.csv"
)

#############
## Model 4 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="SEI", expl_vars=["l_lexical_index_5", "l_pre_demo_ineq_best_SEI", "l_latent_gdppc_mean_log", "l_lexical_index_5_x_l_pre_demo_ineq_best_SEI", "year", "country_id"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 5 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="grg", expl_vars=["l_lexical_index_5", "l_pre_demo_ineq_best_SEI", "l_latent_gdppc_mean_log", "l_lexical_index_5_x_l_pre_demo_ineq_best_SEI", "year", "country_id"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 6 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="ggini", expl_vars=["l_lexical_index_5", "l_pre_demo_ineq_best_SEI", "l_latent_gdppc_mean_log", "l_lexical_index_5_x_l_pre_demo_ineq_best_SEI", "year", "country_id"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

##################
## Mueller 2024 ##
##################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/AJPS/Mueller 2024/m2024_full.csv", encoding='latin-1'
)

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="concession_delivered", expl_vars=["word2vec_1_all_para_normalized", "concession_promised", "size_factor", "violence", "intattention", "e_migdppcln", "v2x_polyarchy", "country_factor"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

####################################
## Baldwin and Ricard-Huguet 2022 ##
####################################

# Load in the dataset using pandas
pol_data = pd.read_stata(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/JOP/Baldwin and Ricard-Huguet 2022/dataset.dta"
)

#############
## Model 1 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="q65", expl_vars=["landqual", "v33", "v5", "country1"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 2 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="TLland", expl_vars=["landqual", "v33", "v5", "country1"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 3 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="TLdisputes", expl_vars=["landqual", "v33", "v5", "country1"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

##############################
## Baturo and Tolstrup 2024 ##
##############################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/JOP/Baturo and Tolstrup 2024/bt2024_full.csv"
)

#############
## Model 2 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="lss_econ", expl_vars=["autocracy", "preselec", "referendum", "firstterm", "gdppcgrowth", "maxnumprotests", "maxsanction", "loggdp", "midb_event", "colorrev", "year"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 5 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="lss_nation", expl_vars=["autocracy", "preselec", "referendum", "firstterm", "gdppcgrowth", "maxnumprotests", "maxsanction", "loggdp", "midb_event", "colorrev", "year"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 8 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="lss_fight", expl_vars=["autocracy", "preselec", "referendum", "firstterm", "gdppcgrowth", "maxnumprotests", "maxsanction", "loggdp", "midb_event", "colorrev", "year"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

###################
## Carcelli 2023 ##
###################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/JOP/Carcelli 2023/all1.csv"
)

#############
## Model 5 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="frag", expl_vars=["year", "countryname", "pluralty", "checks", "lme", "gov1vote", "opp1vote", "herfgov", "frac", "legelec", "exelec", "allhouse", "polariz", "conservatism", "gdp2", "pcap2", "total2"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results


#########################
## Carnegie et al 2024 ##
#########################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/JOP/Carnegie et al 2024/cck2024_full.csv"
)

#############
## Model 2 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="log1p_countsp", expl_vars=["fpopulism", "gdppc", "polity2", "absidealdiff", "execrlc", "dservgni", "ccode", "year"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 3 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="proginit", expl_vars=["fpopulism", "gdppc", "polity2", "absidealdiff", "execrlc", "dservgni", "ccode", "year"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

########################
## Choulis et al 2024 ##
########################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/JOP/Choulis et al 2024/cefm2024_full.csv"
)

#############
## Model 3 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="mean5", expl_vars=["secretpol_revised", "l_ln_pop", "l_ln_gdppc", "l_lexclpop", "l12gr", "nbr_mean5", "intrastate", "ccode", "year"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#########################
## Rogowski et al 2022 ##
#########################

# Load in the dataset using pandas
pol_data = pd.read_stata(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/AJPS/Rogowski et al 2022/country_panel.dta"
)

#############
## Model 2 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="e_migdpgro_5yr", expl_vars=["upu_totalpo_ipo_ln_stock_1_5yr", "e_migdppcln_5yr", "e_mipopula_ipo_ln", "e_miurbaniz_ipo", "e_polity2_ipo", "year", "country_id"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

########################
## Bokobza et al 2022 ##
########################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/JOP/Bokobza et al 2022/bknsa2024_full.csv"
)

#############
## Model 7 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="replacement_rateadj_minister", expl_vars=["coupattempt_whogov", "YEAR_DATA", "L_gdp_cap_pwt_ln", "L_pop_pwt_ln", "L_military_c", "L_monarchy_c", "L_party_c", "L_elec_all", "L_t1_e_a", "L_oil_valuepop_2014_ln", "L_growth_pwt", "L_onset1", "L_nonvio_camp_2", "L_mid_onset", "L_war_onset", "L_domestic3_10", "replacement_rateadj_minister_300", "replacement_rateadj_minister_200", "replacement_rateadj_minister_100", "YEAR_DATA", "COUNTRY_ID"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

##################
## Redeker 2022 ##
##################

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/JOP/Redeker 2022/Cross_nationa_data.csv"
)

#############
## Model 2 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="std_net_lending", expl_vars=["ud", "rti_score", "FDI_standardized", "realgdpgr", "realinterest", "stock_market", "old_age", "cit", "country"]
)  # You must specify the outcome variable


# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

#############
## Model 5 ##
#############

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="std_net_lending", expl_vars=["emprot_reg", "rti_score", "FDI_standardized", "realgdpgr", "realinterest", "stock_market", "old_age", "cit", "country"]
)  # You must specify the outcome variable

# Define the test object
pol_test = CIMissTest(
    pol_dataset
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results

## End
