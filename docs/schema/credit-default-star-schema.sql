CREATE TABLE "fct_company_credit_profile" (
  "company_profile_key" varchar PRIMARY KEY,
  "company_key" varchar NOT NULL,
  "company_id" bigint NOT NULL,
  "snapshot_date" date NOT NULL,
  "company_age_days" integer NOT NULL,
  "active_contracts_count" integer NOT NULL,
  "has_active_electricity_contract" boolean NOT NULL,
  "has_active_gas_contract" boolean NOT NULL,
  "leverage_ratio" decimal(15,4),
  "cash_to_debt_ratio" decimal(15,4),
  "net_profit_margin" decimal(15,4),
  "ebitda" decimal(15,2),
  "max_dpd_trailing_90d" integer NOT NULL,
  "avg_dpd_trailing_90d" decimal(10,2) NOT NULL,
  "unpaid_ratio_trailing_90d" decimal(5,4) NOT NULL,
  "total_outstanding_debt" decimal(15,2) NOT NULL,
  "days_since_last_login" integer,
  "login_velocity" decimal(10,4),
  "billing_disputes_count" integer NOT NULL,
  "average_satisfaction_score" decimal(3,2),
  "is_insolvent" integer NOT NULL
);

CREATE TABLE "dim_companies" (
  "company_key" varchar PRIMARY KEY,
  "company_id" bigint NOT NULL,
  "vat_number" varchar(11) NOT NULL,
  "legal_name" varchar NOT NULL,
  "legal_form" varchar(50) NOT NULL,
  "foundation_date" date NOT NULL,
  "is_active" bool NOT NULL,
  "registered_office_region" varchar(100),
  "industry_sector" varchar(30) NOT NULL,
  "dbt_valid_from" timestamp NOT NULL,
  "dbt_valid_to" timestamp
);

CREATE TABLE "dim_date" (
  "date_day" date PRIMARY KEY,
  "month_number" integer NOT NULL,
  "quarter" integer NOT NULL,
  "year" integer NOT NULL,
  "day_of_week" integer NOT NULL,
  "day_name" varchar(10) NOT NULL,
  "is_weekend" boolean NOT NULL
);

CREATE INDEX ON "dim_companies" ("company_id");

COMMENT ON COLUMN "fct_company_credit_profile"."company_profile_key" IS 'Surrogate primary key (Hash of company_id + snapshot_date)';

COMMENT ON COLUMN "fct_company_credit_profile"."company_key" IS 'Surrogate foreign key linking to the correct historical version in dim_companies resolved at snapshot_date';

COMMENT ON COLUMN "fct_company_credit_profile"."company_id" IS 'Foreign key to dim_companies';

COMMENT ON COLUMN "fct_company_credit_profile"."snapshot_date" IS 'Foreign key to dim_date acting as the point-in-time ML cutoff date';

COMMENT ON COLUMN "fct_company_credit_profile"."company_age_days" IS 'Denormalized feature: snapshot_date - foundation_date. Pre-calculated at mart build time to serve the training/inference export directly, avoiding a downstream feature-engineering step';

COMMENT ON COLUMN "fct_company_credit_profile"."active_contracts_count" IS 'Count of energy contracts with contract_status = ''active'' AS OF snapshot_date (resolved via the companies/energy_contracts SCD2 snapshot history, not the current live status)';

COMMENT ON COLUMN "fct_company_credit_profile"."has_active_electricity_contract" IS 'AS OF snapshot_date, resolved via SCD2 history';

COMMENT ON COLUMN "fct_company_credit_profile"."has_active_gas_contract" IS 'AS OF snapshot_date, resolved via SCD2 history';

COMMENT ON COLUMN "fct_company_credit_profile"."leverage_ratio" IS 'Sourced from FACT_FINANCIAL_STATEMENTS. total_debt / share_capital';

COMMENT ON COLUMN "fct_company_credit_profile"."cash_to_debt_ratio" IS 'Sourced from FACT_FINANCIAL_STATEMENTS. liquidity_cash / total_debt';

COMMENT ON COLUMN "fct_company_credit_profile"."net_profit_margin" IS 'Sourced from FACT_FINANCIAL_STATEMENTS. net_income / total_revenue';

COMMENT ON COLUMN "fct_company_credit_profile"."ebitda" IS 'Sourced from FACT_FINANCIAL_STATEMENTS. EBITDA amount';

COMMENT ON COLUMN "fct_company_credit_profile"."max_dpd_trailing_90d" IS 'Maximum days past due observed on invoices due within the last 90 days';

COMMENT ON COLUMN "fct_company_credit_profile"."avg_dpd_trailing_90d" IS 'Average days past due on invoices due within the last 90 days';

COMMENT ON COLUMN "fct_company_credit_profile"."unpaid_ratio_trailing_90d" IS 'Ratio of unpaid invoices in the last 90 days relative to total invoices issued';

COMMENT ON COLUMN "fct_company_credit_profile"."total_outstanding_debt" IS 'Sum of total_amount of invoices still unpaid/overdue as of snapshot_date';

COMMENT ON COLUMN "fct_company_credit_profile"."days_since_last_login" IS 'Days between snapshot_date and the most recent login_timestamp from FACT_USER_LOGINS';

COMMENT ON COLUMN "fct_company_credit_profile"."login_velocity" IS 'Ratio: logins in trailing 30 days / average monthly logins in trailing 90 days';

COMMENT ON COLUMN "fct_company_credit_profile"."billing_disputes_count" IS 'Number of FACT_CRM_TICKETS in ''billing'' category opened in the last 90 days. Defaults to 0 via COALESCE when no matching tickets exist';

COMMENT ON COLUMN "fct_company_credit_profile"."average_satisfaction_score" IS 'Average of satisfaction scores for tickets resolved in the last 90 days. Nullable: no value when no tickets were resolved in the window';

COMMENT ON COLUMN "fct_company_credit_profile"."is_insolvent" IS 'TARGET VARIABLE. Sourced from FACT_INSOLVENCY_SNAPSHOT. 1 if max_dpd >= 90 days in trailing window, else 0';

COMMENT ON TABLE "dim_companies" IS 'Slowly Changing Dimension Type 2. Resolved against dbt_valid_from/dbt_valid_to
at snapshot_date to give fct_company_credit_profile the attribute values that
were actually true at the point-in-time cutoff, preventing leakage from
attributes (industry_sector, is_active, registered_office_region) that can
change after the observation date.
';

COMMENT ON COLUMN "dim_companies"."company_key" IS 'Surrogate key (Hash of company_id + dbt_valid_from). One row per historical version of the company record';

COMMENT ON COLUMN "dim_companies"."company_id" IS 'Natural/business key. Multiple rows share the same company_id across its version history';

COMMENT ON COLUMN "dim_companies"."is_active" IS 'Value of is_active as it stood during this version''s validity window';

COMMENT ON COLUMN "dim_companies"."industry_sector" IS 'Value of industry_sector as it stood during this version''s validity window';

COMMENT ON COLUMN "dim_companies"."dbt_valid_from" IS 'Start of this version''s validity window';

COMMENT ON COLUMN "dim_companies"."dbt_valid_to" IS 'End of this version''s validity window. NULL if this is the current version';

COMMENT ON COLUMN "dim_date"."date_day" IS 'Calendar date primary key';

ALTER TABLE "fct_company_credit_profile" ADD FOREIGN KEY ("company_key") REFERENCES "dim_companies" ("company_key") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "fct_company_credit_profile" ADD FOREIGN KEY ("snapshot_date") REFERENCES "dim_date" ("date_day") DEFERRABLE INITIALLY IMMEDIATE;
