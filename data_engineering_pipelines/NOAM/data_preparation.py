'''
NOTE
-------
Author | rjaswal@estee.com
Date   | 2024-02-06 
-------
Objective:
- Prepare and preprocess data for NOAM Major Category Affinity model training.
- Perform necessary data cleaning, transformation

Key Functionality:
- Load raw data from specified sources.
- Clean and preprocess data (handling missing values, outliers, etc.).
- Generate and select features relevant for modeling.
- Split data into training and testing sets.
- Save processed data for model training.
'''

import pandas as pd
from pyspark.sql import SparkSession

from pyspark.ml.functions import vector_to_array
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder.appName("ML Azure -- Product Category Affinity").getOrCreate()

# SQL Queries Dictionary
SQL_QUERIES = {
    "create_database": "create database if not exists db_cat_aff;",

    "create_major_category_view": """
        create or replace view db_cat_aff.v_noam_category_affinity_major_category as
        with category as (
            select distinct sku_six_digit_identifier, major_category_name
            from db_fnd_gold.product_dim
            where major_category_name is not null and product_type_code = 'FERT'
        ), smy as (
            select sku_six_digit_identifier, major_category_name, count(*) over(partition by sku_six_digit_identifier) cnt
            from category
        )
        select sku_six_digit_identifier, major_category_name
        from smy
        where cnt=1
    """,

    "create_brand_lookup_view": """
        create or replace view db_cat_aff.v_category_affinity_brand_lkup_noam as
        select brand_id, brand_cd, brand_desc, brand_name
        from db_fnd_bronze.v_brand_lkup_noam b
        where cdp_region = 'noam'
          and b.brand_cd not in ('CS', 'PR', 'CT')
    """,

    "create_transaction_header_view": """
        create or replace view db_cat_aff.v_noam_category_affinity_txn_header as
        select elc_master_id, consumer_id, brand_id, sale_dt, txn_id, usd_net_order_total, currency_cd, channel_cd,
               usd_product_total, usd_product_discount_total, usd_returns_total, loyalty_trans_flag, store_id, store_num
        from db_fnd_bronze.v_consumer_transaction_header_noam
        where cdp_region = 'noam' and elc_master_id is not null 
              and consumer_id IS NOT NULL 
              and valid_ind = 1 
              and consumer_valid_flag = 'Y' 
              and mac_pro_flag = 'N' 
              and usd_net_order_total > 0 
              and currency_cd in ('USD', 'CAD') 
              and visit_ind = 1
              and CDC_CODE <> 'D'
              and sale_dt > '2013-07-01'
    """,

    "create_transaction_detail_view": """
        create or replace view db_cat_aff.v_noam_category_affinity_txn_detail as
        select sale_dt, channel_cd, elc_master_id, consumer_id, brand_id, 
               txn_id, txn_detail_id, sale_line_type, sale_line_num, src_txn_cd_desc, 
               net_item_qty, usd_total_price, usd_unit_price, 
               product_id, product_sku, product_desc, 
               master_category, category, sub_category, subcategory_alter
        from db_fnd_bronze.v_consumer_transaction_detail_noam
        where cdp_region = 'noam' 
              and elc_master_id is not null 
              and consumer_id is not null
              and product_sku is not null
              and sale_line_type ='Sale' and net_item_qty > 0
              and sale_dt > '2013-07-01'
    """,

    "create_sales_details_view": """
        create or replace view db_cat_aff.v_noam_category_affinity_sales_details as
        select d.sale_dt, d.channel_cd, d.txn_id, 
               d.elc_master_id, d.consumer_id, d.brand_id, 
               c.major_category_name as major_category, 
               h.usd_net_order_total header_sales_amount,
               txn_detail_id, sale_line_type, sale_line_num, src_txn_cd_desc, 
               net_item_qty, usd_total_price sales_amount, usd_unit_price, 
               product_id, product_sku, product_desc, 
               master_category, category, sub_category, subcategory_alter
        from db_cat_aff.v_noam_category_affinity_txn_detail d
        inner join db_cat_aff.v_noam_category_affinity_txn_header h on d.txn_id = h.txn_id
        inner join db_cat_aff.v_noam_category_affinity_major_category c on substr(d.product_sku, 1, 6) = c.sku_six_digit_identifier
        inner join db_cat_aff.v_category_affinity_brand_lkup_noam b on d.brand_id = b.brand_id
    """,

    "create_sales_monthly_summary_table": """
        create table if not exists db_cat_aff.sales_monthly_summary_noam (
          sale_month date,
          etl_date timestamp,
          elc_master_id string,
          consumer_id bigint,
          brand_id bigint,
          sales_amount double,
          txn_cnt bigint,
          item_cnt bigint,
          makeup_amount double,
          skincare_amount double,
          fragrance_amount double,
          haircare_amount double,
          other_amount double
        ) partitioned by (sale_month)
    """,

    "insert_sales_monthly_summary": """
        insert into db_cat_aff.sales_monthly_summary_noam partition(sale_month)
        select to_date(date_trunc('MONTH', sale_dt)) AS sale_month,
               now() as etl_date,
               elc_master_id, consumer_id, brand_id, 
               sum(sales_amount) as sales_amount,
               count(distinct txn_id) as txn_cnt,
               count(*) as item_cnt,  
               sum(case when major_category = 'Makeup' then sales_amount else 0 end) as makeup_amount,
               sum(case when major_category = 'Skincare' then sales_amount else 0 end) as skincare_amount,
               sum(case when major_category = 'Fragrance' then sales_amount else 0 end) as fragrance_amount,
               sum(case when major_category = 'Haircare' then sales_amount else 0 end) as haircare_amount,
               sum(case when major_category = 'Other' then sales_amount else 0 end) as other_amount
        from db_cat_aff.v_noam_category_affinity_sales_details
        where sale_dt < '2024-01-01'
        group by sale_month, elc_master_id, consumer_id, brand_id
    """,

    "create_sales_snapshot_view": """
        create or replace view db_cat_aff.v_sales_summary as 
        with calendar as (
            select explode(sequence(to_date('2013-07-01'), date_sub(date_trunc('MONTH',current_date()), 1), interval 1 month)) as dt
        ), cst_base as (
            select elc_master_id, consumer_id, brand_id, min(sale_month) min_sale_month 
            from db_cat_aff.sales_monthly_summary_noam
            group by elc_master_id, consumer_id, brand_id
        ), cst_cal as (
            select elc_master_id, consumer_id, brand_id, min_sale_month, dt 
            from cst_base cross join calendar 
            where dt>=min_sale_month
        ), cst_smy_month as (
            select c.dt, c.elc_master_id, c.consumer_id, c.brand_id, s.sale_month, s.sales_amount, s.txn_cnt, s.item_cnt, 
                   s.makeup_amount, s.skincare_amount, s.fragrance_amount, s.haircare_amount, s.other_amount
            from cst_cal c
            left join db_cat_aff.sales_monthly_summary_noam s on c.consumer_id = s.consumer_id and c.dt = s.sale_month
            and c.elc_master_id = s.elc_master_id and c.brand_id = s.brand_id
    ),
    final as (
    select      to_date(dt) as snapshot_dt, 
                elc_master_id, consumer_id, 
                first_value(brand_id) over(partition by elc_master_id, consumer_id, brand_id) as brand_id,

                -- target definition
                case when (sum(makeup_amount) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 1 following and 3 following)) > 0
                then 1 else 0 end as target_makeup,
                case when (sum(skincare_amount)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 1 following and 3 following)) > 0
                then 1 else 0 end as target_skincare,
                case when (sum(fragrance_amount)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 1 following and 3 following)) > 0
                then 1 else 0 end as target_fragrance,
                case when (sum(haircare_amount)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 1 following and 3 following)) > 0
                then 1 else 0 end as target_haircare, 

                -- global stats            
                min(sale_month) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row)
                as first_purchase_dt,
                max(sale_month)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row)
                as last_purchase_dt,            
                sum(sales_amount) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as sales_tot, 
                sum(txn_cnt) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as txn_cnt_tot,
                sum(case when sales_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 11 preceding and current row) 
                as sales_mnt_cnt_12m,            
                avg(case when item_cnt>0 then item_cnt else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt 
                rows between unbounded preceding and current row) 
                as avg_item_cnt,

                -- last month
                makeup_amount as makep_amt_last, 
                skincare_amount as skincare_amt_last, 
                fragrance_amount as fragrance_amt_last, 
                haircare_amount as haircare_amt_last, 
                other_amount as other_amt_last,
                item_cnt item_cnt_last, 
                txn_cnt as txn_cnt_last, 
                sales_amount as sales_amt_last,

                -- makeup lagging
                sum(case when makeup_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 2 preceding and current row) 
                as makeup_cnt_3m,
                sum(case when makeup_amount>0 then 1 else 0 end)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 5 preceding and current row) 
                as makeup_cnt_6m,
                sum(case when makeup_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 11 preceding and current row) 
                as makeup_cnt_12m,
                sum(case when makeup_amount>0 then 1 else 0 end)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 23 preceding and current row) 
                as makeup_cnt_24m,
                avg(case when makeup_amount>0 then makeup_amount else null end)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 11 preceding and current row) 
                as makeup_avg_12m,

                -- makeup global
                sum(makeup_amount) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row)
                as makeup_amt_tot,
                sum(case when makeup_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as makeup_cnt_tot,            
                min(case when makeup_amount>0 then makeup_amount else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as makeup_min,
                max(case when makeup_amount>0 then makeup_amount else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as makeup_max,
                avg(case when makeup_amount>0 then makeup_amount else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row)
                as makeup_avg,   

                -- skincare lagging
                sum(case when skincare_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 2 preceding and current row) 
                as skincare_cnt_3m,
                sum(case when skincare_amount>0 then 1 else 0 end)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 5 preceding and current row) 
                as skincare_cnt_6m,
                sum(case when skincare_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 11 preceding and current row) 
                as skincare_cnt_12m,
                sum(case when skincare_amount>0 then 1 else 0 end)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 23 preceding and current row) 
                as skincare_cnt_24m,
                avg(case when skincare_amount>0 then skincare_amount else null end)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 11 preceding and current row) 
                as skincare_avg_12m,

                -- skincare global
                sum(skincare_amount) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as skincare_amt_tot,
                sum(case when skincare_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as skincare_cnt_tot,            
                min(case when skincare_amount>0 then skincare_amount else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as skincare_min,
                max(case when skincare_amount>0 then skincare_amount else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as skincare_max,
                avg(case when skincare_amount>0 then skincare_amount else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row)
                as skincare_avg,

                -- fragrance lagging
                sum(case when fragrance_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 2 preceding and current row) 
                as fragrance_cnt_3m,
                sum(case when fragrance_amount>0 then 1 else 0 end)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 5 preceding and current row) 
                as fragrance_cnt_6m,
                sum(case when fragrance_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 11 preceding and current row) 
                as fragrance_cnt_12m,
                sum(case when fragrance_amount>0 then 1 else 0 end)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 23 preceding and current row) 
                as fragrance_cnt_24m,
                avg(case when fragrance_amount>0 then fragrance_amount else null end)
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between 11 preceding and current row) 
                as fragrance_avg_12m,

                -- fragrance global
                sum(fragrance_amount) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as fragrance_amt_tot,
                sum(case when fragrance_amount>0 then 1 else 0 end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as fragrance_cnt_tot,            
                min(case when fragrance_amount>0 then fragrance_amount else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as fragrance_min,
                max(case when fragrance_amount>0 then fragrance_amount else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row) 
                as fragrance_max,
                avg(case when fragrance_amount>0 then fragrance_amount else null end) 
                over(partition by elc_master_id, consumer_id, brand_id  order by dt
                rows between unbounded preceding and current row)
                as fragrance_avg
    from        cst_smy_month
    )
    select      f.*,
                months_between(snapshot_dt, first_purchase_dt) as months_since_first_purchase,
                months_between(snapshot_dt, last_purchase_dt) months_since_last_purchase
    from        final f
    --where       snapshot_dt = '2021-12-01'
    ;
    """,

    "create_sales_snapshot_table": """
        create table if not exists db_cat_aff.sales_snapshot_noam (
            snapshot_dt date,
            elc_master_id string,
            consumer_id bigint,
            brand_id bigint,
            target_makeup int,
            target_skincare int,
            target_fragrance int,
            target_haircare int,
            first_purchase_dt date,
            last_purchase_dt date,
            sales_tot double,
            txn_cnt_tot double,
            sales_mnt_cnt_12m double,
            avg_item_cnt double,
            makep_amt_last double,
            skincare_amt_last double,
            fragrance_amt_last double,
            haircare_amt_last double,
            other_amt_last double,
            item_cnt_last double,
            txn_cnt_last double,
            sales_amt_last double,
            makeup_cnt_3m double,
            makeup_cnt_6m double,
            makeup_cnt_12m double,
            makeup_cnt_24m double,
            makeup_avg_12m double,
            makeup_amt_tot double,
            makeup_cnt_tot double,
            makeup_min double,
            makeup_max double,
            makeup_avg double,
            skincare_cnt_3m double,
            skincare_cnt_6m double,
            skincare_cnt_12m double,
            skincare_cnt_24m double,
            skincare_avg_12m double,
            skincare_amt_tot double,
            skincare_cnt_tot double,
            skincare_min double,
            skincare_max double,
            skincare_avg double,
            fragrance_cnt_3m double,
            fragrance_cnt_6m double,
            fragrance_cnt_12m double,
            fragrance_cnt_24m double,
            fragrance_avg_12m double,
            fragrance_amt_tot double,
            fragrance_cnt_tot double,
            fragrance_min double,
            fragrance_max double,
            fragrance_avg double,
            months_since_first_purchase double,
            months_since_last_purchase double
        ) partitioned by (snapshot_dt)
    """,

    "create_custom_calculated_attributes_table": """
        create table if not exists db_cat_aff.custom_calculated_attributes_noam (
            snapshot_dt date,
            elc_master_id string,
            consumer_id bigint,
            brand_id bigint, 
            fss_txn_ratio double,
            mobile_txn_ratio double,
            web_txn_ratio double,
            tablet_txn_ratio double,
            discount_ratio double,
            sales_q4_ratio double,
            loyalty_txn_ratio double,
            return_txn_ratio double
        ) partitioned by (snapshot_dt)
    """,

    "create_store_calculated_attributes_table": """
        create table if not exists db_cat_aff.store_calculated_attributes_noam (
            snapshot_dt date,
            elc_master_id string,
            consumer_id bigint,
            brand_id bigint, 
            most_freq_store_id bigint,
            store_channel string,
            store_market_cd string,
            store_name string,
            store_brand_cd string,
            store_region string,
            store_sub_region string,
            store_state_prov_cd string,
            store_city string,
            store_postal_cd string,
            store_latitude double,
            store_longitude double,
            store_status_cd string
        ) partitioned by (snapshot_dt)
    """,

    "create_pdt_emp_lkp_view": """
        create or replace view db_cat_aff.v_noam_category_affinity_pdt_emp_lkp as
        select      dt.*, hash(pdt_clean) as pdt_hash from(
        select      distinct 
                    p.sku_six_digit_identifier, 
                    lower(REGEXP_REPLACE(
                    REGEXP_REPLACE(major_category_name ||'_'|| p.category_name ||'_'|| p.sub_category_name,
                                    '[&,/ ()-]', '_'),
                                    '_+', '_'))
                    as pdt_clean
        from        db_fnd_gold.product_dim p
        where       major_category_name is not null 
                and product_type_code = 'FERT'
        ) as dt
    """,

    "create_emb_details_view": """
        create or replace view db_cat_aff.v_noam_category_affinity_emb_details as
        select      d.sale_dt, d.txn_id, d.elc_master_id, d.consumer_id, c.pdt_clean, c.pdt_hash,
                    txn_detail_id, sale_line_type, sale_line_num, src_txn_cd_desc, 
                    net_item_qty, usd_total_price sales_amount, usd_unit_price, 
                    product_id, product_sku, product_desc
        from        db_cat_aff.v_noam_category_affinity_txn_detail d
        inner join  db_cat_aff.v_noam_category_affinity_txn_header h on d.txn_id = h.txn_id
        inner join  db_cat_aff.v_noam_category_affinity_pdt_emp_lkp c on substr(d.product_sku, 1, 6) = c.sku_six_digit_identifier
        inner join  db_cat_aff.v_category_affinity_brand_lkup_noam b on d.brand_id = b.brand_id
    """,

    "create_master_table_view": """
        create or replace view db_cat_aff.v_master_table as  
        select d.*, 
            cv.ethnic_group,
            cv.gender,
            cv.language,
            cv.marital_status,
            cv.occupation_group,
            cv.mosaic_segment,
            cv.mosaic_group,
            cv.age_range,
            cv.ethnicity_detail,
            cv.income,
            date_part('year', d.snapshot_dt)-cv.birth_year age,
            case when ca.consumer_id is not null then 'Y' else 'N' end as exists_in_cdp,
            ca.country_cd,
            ca.state_cd,
            ca.lang_cd,
            cast(ca.email_opt_in_ind as int) as email_opt_in_ind,
            cast(ca.phone_opt_in_ind as int) as phone_opt_in_ind,
            ca.city,
            ca.zip_code,
            cast(ca.dm_opt_in_ind as int) as dm_opt_in_ind,
            cast(ca.mobile_contactable_ind as int) as mobile_contactable_ind,
            cast(ca.first_store_id as int) as first_store_id,
            cast(ca.last_store_id as int) as last_store_id,
            cast(ca.closest_store as int) as closest_store,
            cast(ca.most_frequent_store as int) as most_frequent_store,
            ca.first_channel,
            ca.deceased_flag,
            ca.prison_flag,
            cast(ca.mobile_opt_in_ind as int) as mobile_opt_in_ind,
            ca.dm_state,
            ca.dm_country,
            cac.fss_txn_ratio,
            cac.mobile_txn_ratio,
            cac.web_txn_ratio,
            cac.tablet_txn_ratio,
            cac.discount_ratio,
            cac.sales_q4_ratio,
            cac.loyalty_txn_ratio,
            cac.return_txn_ratio,
            cas.most_freq_store_id,
            cas.store_channel,
            cas.store_market_cd,
            cas.store_name,
            cas.store_brand_cd,
            cas.store_region,
            cas.store_sub_region,
            cas.store_state_prov_cd,
            cas.store_city,
            cas.store_postal_cd,
            cas.store_latitude,
            cas.store_longitude,
            cas.store_status_cd
        from db_cat_aff.sales_snapshot_noam d
        left join   db_fnd_bronze.v_consumer_custom_cv_vars_noam as cv
                on  d.elc_master_id = cv.elc_master_id 
                and cv.cdp_region = 'noam' 
                and cv.CDC_CODE <> 'D'
        left join   db_fnd_bronze.v_consumer_calculated_attributes_noam ca
                on  d.elc_master_id = ca.elc_master_id
                and d.consumer_id = ca.consumer_id 
                and ca.cdp_region = 'noam' 
                --and ca.CDC_CODE <> 'D'
        left join   db_cat_aff.custom_calculated_attributes_noam cac 
                on  d.consumer_id = cac.consumer_id
                and d.elc_master_id = cac.elc_master_id
                and d.brand_id = cac.brand_id
                and d.snapshot_dt = cac.snapshot_dt
        left join   db_cat_aff.store_calculated_attributes_noam cas
                on  d.consumer_id = cas.consumer_id
                and d.elc_master_id = cas.elc_master_id
                and d.brand_id = cas.brand_id
                and d.snapshot_dt = cas.snapshot_dt
    """,

    "create_master_table": """
        create table if not exists db_cat_aff.master_table_noam  using delta
        partitioned by (snapshot_dt) 
        as select * from db_cat_aff.v_master_table where 1=0
    """
}

def execute_sql_query(query_key, spark):
    """
    Executes an SQL query based on the provided key.

    Args:
        query_key (str): The key corresponding to the SQL query in the SQL_QUERIES dictionary.
        spark (SparkSession): The SparkSession object used to execute the query.

    Returns:
        DataFrame: The result of the SQL query execution.
    """
    query = SQL_QUERIES.get(query_key)
    if query is None:
        raise ValueError(f"Query key '{query_key}' not found in SQL_QUERIES dictionary.")
    return spark.sql(query)

class DataPreparationPipeline:
    def __init__(self, spark):
        self.spark = spark

    def run(self):
        """Executes the data preparation steps."""
        self.init_database()
        self.create_major_category_view()
        self.create_brand_lookup_view()
        self.create_transaction_header_view()
        self.create_transaction_detail_view()
        self.create_sales_details_view()
        self.create_sales_monthly_summary_table()
        self.insert_sales_monthly_summary()
        self.create_sales_snapshot_view()
        self.create_sales_snapshot_table()
        self.create_custom_calculated_attributes_table()
        self.create_store_calculated_attributes_table()
        self.create_master_table_view()
        self.create_master_table()
        self.prepare_consumer_baskets()


    def init_database(self):
        """Initializes the database."""
        execute_sql_query("create_database", self.spark)

    def create_major_category_view(self):
        """Creates the major category view."""
        execute_sql_query("create_major_category_view", self.spark)

    def create_brand_lookup_view(self):
        """Creates the brand lookup view."""
        execute_sql_query("create_brand_lookup_view", self.spark)

    def create_transaction_header_view(self):
        """Creates the transaction header view."""
        execute_sql_query("create_transaction_header_view", self.spark)

    def create_transaction_detail_view(self):
        """Creates the transaction detail view."""
        execute_sql_query("create_transaction_detail_view", self.spark)

    def create_sales_details_view(self):
        """Creates the sales details view."""
        execute_sql_query("create_sales_details_view", self.spark)

    def create_sales_monthly_summary_table(self):
        """Creates the sales monthly summary table."""
        execute_sql_query("create_sales_monthly_summary_table", self.spark)

    def insert_sales_monthly_summary(self):
        """Inserts data into the sales monthly summary table."""
        execute_sql_query("insert_sales_monthly_summary", self.spark)

    def create_sales_snapshot_view(self):
        """Creates the sales snapshot view."""
        execute_sql_query("create_sales_snapshot_view", self.spark)

    def create_sales_snapshot_table(self):
        """Creates the sales snapshot table."""
        execute_sql_query("create_sales_snapshot_table", self.spark)

    def create_custom_calculated_attributes_table(self):
        """Creates the custom calculated attributes table."""
        execute_sql_query("create_custom_calculated_attributes_table", self.spark)

    def create_store_calculated_attributes_table(self):
        """Creates the store calculated attributes table."""
        execute_sql_query("create_store_calculated_attributes_table", self.spark)

    def create_master_table_view(self):
        """Creates the master table view."""
        execute_sql_query("create_master_table_view", self.spark)

    def create_master_table(self):
        """Creates the master table."""
        execute_sql_query("create_master_table", self.spark)

    def prepare_consumer_baskets(self):
        """
        Prepares consumer baskets by aggregating product hashes for each consumer.
        """
        # Assuming 'vw_emb_det' corresponds to 'db_cat_aff.v_noam_category_affinity_emb_details' view
        threshold_date = "2022-04-01"
        consumer_basket = self.spark.sql(f"""
            SELECT consumer_id, COLLECT_LIST(cast(pdt_hash as string)) AS basket
            FROM db_cat_aff.v_noam_category_affinity_emb_details
            WHERE sale_dt < '{threshold_date}'
            GROUP BY consumer_id
        """)
        consumer_basket.write.mode('overwrite').saveAsTable('db_cat_aff.emb_basket')

# Main
if __name__ == "__main__":
    pipeline = DataPreparationPipeline(spark)
    pipeline.run()