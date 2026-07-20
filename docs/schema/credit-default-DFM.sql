CREATE TABLE "FACT_BILLING_TRANSACTIONS" (
  "issue_date" date,
  "due_date" date,
  "payment_date" date,
  "contract_id" bigint,
  "commodity_type" varchar,
  "billing_status_key" bigint,
  "invoice_number" varchar,
  "transaction_reference" varchar,
  "electricity_consumption_kwh" decimal,
  "gas_consumption_scm" decimal,
  "amount_excluding_tax" decimal,
  "tax_amount" decimal,
  "total_amount" decimal,
  "amount_paid" decimal,
  "payment_term_days" integer,
  "days_past_due" integer
);

CREATE TABLE "FACT_CRM_TICKETS" (
  "created_at" timestamptz,
  "resolved_at" timestamptz,
  "company_id" bigint,
  "ticket_category" varchar,
  "satisfaction_score" integer,
  "resolution_time_hours" decimal,
  "resolution_time_days" decimal
);

CREATE TABLE "FACT_USER_LOGINS" (
  "login_timestamp" timestamptz,
  "company_id" bigint,
  "device_type" varchar,
  "login_count" integer
);

CREATE TABLE "FACT_FINANCIAL_STATEMENTS" (
  "company_id" bigint,
  "fiscal_year" integer,
  "total_revenue" decimal,
  "net_income" decimal,
  "total_debt" decimal,
  "liquidity_cash" decimal,
  "share_capital" decimal,
  "ebitda" decimal,
  "leverage_ratio" decimal,
  "cash_to_debt_ratio" decimal,
  "net_profit_margin" decimal,
  PRIMARY KEY ("company_id", "fiscal_year")
);

CREATE TABLE "FACT_INSOLVENCY_SNAPSHOT" (
  "company_id" bigint,
  "snapshot_date" date,
  "is_insolvent" integer,
  "max_dpd_trailing_window" integer,
  PRIMARY KEY ("company_id", "snapshot_date")
);

CREATE TABLE "DIM_DATE" (
  "date_day" date PRIMARY KEY,
  "month_number" integer,
  "quarter" integer,
  "year" integer
);

CREATE TABLE "DIM_ENERGY_CONTRACTS" (
  "contract_id" bigint PRIMARY KEY,
  "company_id" bigint,
  "commodity_type" varchar,
  "market_type" varchar,
  "voltage_level" varchar,
  "pressure_level" varchar,
  "power_committed_kw" decimal,
  "gas_meter_class" varchar,
  "activation_date" date,
  "termination_date" date,
  "contract_status" varchar,
  "contract_duration_days" integer
);

CREATE TABLE "DIM_COMPANIES" (
  "company_id" bigint PRIMARY KEY,
  "vat_number" varchar,
  "legal_name" varchar,
  "legal_form" varchar,
  "foundation_date" date,
  "company_age_days" integer,
  "is_active" bool,
  "registered_office_region" varchar,
  "industry_sector" varchar
);

CREATE TABLE "DIM_GEOGRAPHY" (
  "registered_office_region" varchar PRIMARY KEY
);

CREATE TABLE "DIM_SECTOR" (
  "industry_sector" varchar PRIMARY KEY
);

CREATE TABLE "DIM_BILLING_STATUS_JUNK" (
  "billing_status_key" bigint PRIMARY KEY,
  "invoice_status" varchar,
  "payment_method" varchar,
  "payment_status" varchar
);

CREATE TABLE "DIM_DEVICE" (
  "device_type" varchar PRIMARY KEY
);

CREATE TABLE "DIM_TICKET_CATEGORY" (
  "ticket_category" varchar PRIMARY KEY
);

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."billing_status_key" IS 'Link to DIM_BILLING_STATUS_JUNK (invoice_status, payment_method, payment_status)';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."invoice_number" IS 'Unique invoice identifier. Sourced from invoices.invoice_number';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."transaction_reference" IS 'Unique payment gateway transaction identifier. Sourced from payments.transaction_reference';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."electricity_consumption_kwh" IS 'Sourced from stg_invoices. Null for gas contracts';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."gas_consumption_scm" IS 'Sourced from stg_invoices. Null for electricity contracts';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."amount_excluding_tax" IS 'Net billable amount';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."tax_amount" IS 'Calculated tax value';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."total_amount" IS 'Gross total invoice amount (Net + Tax)';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."amount_paid" IS 'Sourced from stg_payments. Total settled amount';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."payment_term_days" IS 'Constructed feature: due_date - issue_date';

COMMENT ON COLUMN "FACT_BILLING_TRANSACTIONS"."days_past_due" IS 'Constructed feature: payment_date - due_date (if paid late) or current_date - due_date (if unpaid)';

COMMENT ON COLUMN "FACT_CRM_TICKETS"."satisfaction_score" IS 'Customer CSAT rating (1 to 5)';

COMMENT ON COLUMN "FACT_CRM_TICKETS"."resolution_time_hours" IS 'Constructed feature: delta between resolved_at and created_at in hours';

COMMENT ON COLUMN "FACT_CRM_TICKETS"."resolution_time_days" IS 'Constructed feature: delta between resolved_at and created_at in days';

COMMENT ON COLUMN "FACT_USER_LOGINS"."login_count" IS 'Semi-additive measure representing the frequency of access events';

COMMENT ON COLUMN "FACT_FINANCIAL_STATEMENTS"."leverage_ratio" IS 'Constructed ratio: total_debt / share_capital';

COMMENT ON COLUMN "FACT_FINANCIAL_STATEMENTS"."cash_to_debt_ratio" IS 'Constructed ratio: liquidity_cash / total_debt';

COMMENT ON COLUMN "FACT_FINANCIAL_STATEMENTS"."net_profit_margin" IS 'Constructed ratio: net_income / total_revenue';

COMMENT ON COLUMN "FACT_INSOLVENCY_SNAPSHOT"."snapshot_date" IS 'Observation date for the insolvency label. Defines the point-in-time cutoff used to avoid data leakage';

COMMENT ON COLUMN "FACT_INSOLVENCY_SNAPSHOT"."is_insolvent" IS 'TARGET VARIABLE (Y). Business definition: max_dpd >= 90 days, computed over the trailing observation window ending at snapshot_date';

COMMENT ON COLUMN "FACT_INSOLVENCY_SNAPSHOT"."max_dpd_trailing_window" IS 'Constructed feature: max(days_past_due) from FACT_BILLING_TRANSACTIONS over the trailing window ending at snapshot_date';

COMMENT ON COLUMN "DIM_ENERGY_CONTRACTS"."commodity_type" IS 'Electricity or Gas';

COMMENT ON COLUMN "DIM_ENERGY_CONTRACTS"."market_type" IS 'Regulated or Deregulated';

COMMENT ON COLUMN "DIM_ENERGY_CONTRACTS"."voltage_level" IS 'Low, Medium, or High';

COMMENT ON COLUMN "DIM_ENERGY_CONTRACTS"."pressure_level" IS 'Low, Medium, or High';

COMMENT ON COLUMN "DIM_ENERGY_CONTRACTS"."contract_status" IS 'Active, Suspended, or Terminated';

COMMENT ON COLUMN "DIM_ENERGY_CONTRACTS"."contract_duration_days" IS 'Constructed feature: lifetime of contract in days';

COMMENT ON COLUMN "DIM_COMPANIES"."vat_number" IS 'Unique 11-digit corporate tax identifier';

COMMENT ON COLUMN "DIM_COMPANIES"."legal_form" IS 'Legal entity type (e.g., SRL, SPA, SRLS)';

COMMENT ON COLUMN "DIM_COMPANIES"."company_age_days" IS 'Constructed feature: current_date - foundation_date';

COMMENT ON TABLE "DIM_BILLING_STATUS_JUNK" IS 'Junk dimension collapsing the low-cardinality enum attributes of FACT_BILLING_TRANSACTIONS
(invoice_status, payment_method, payment_status) into a single dimension table, one row
per observed combination, to avoid three separate near-trivial dimension tables.
';

COMMENT ON COLUMN "DIM_BILLING_STATUS_JUNK"."billing_status_key" IS 'Surrogate key for the combination of low-cardinality status/method flags below';

COMMENT ON COLUMN "DIM_BILLING_STATUS_JUNK"."invoice_status" IS 'Domain: unpaid, paid, overdue, cancelled';

COMMENT ON COLUMN "DIM_BILLING_STATUS_JUNK"."payment_method" IS 'Domain: direct_debit, bank_transfer, credit_card, postal_bulletin, cash';

COMMENT ON COLUMN "DIM_BILLING_STATUS_JUNK"."payment_status" IS 'Domain: pending, completed, failed, refunded';

ALTER TABLE "FACT_BILLING_TRANSACTIONS" ADD FOREIGN KEY ("issue_date") REFERENCES "DIM_DATE" ("date_day") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_BILLING_TRANSACTIONS" ADD FOREIGN KEY ("due_date") REFERENCES "DIM_DATE" ("date_day") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_BILLING_TRANSACTIONS" ADD FOREIGN KEY ("payment_date") REFERENCES "DIM_DATE" ("date_day") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_BILLING_TRANSACTIONS" ADD FOREIGN KEY ("contract_id") REFERENCES "DIM_ENERGY_CONTRACTS" ("contract_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_BILLING_TRANSACTIONS" ADD FOREIGN KEY ("billing_status_key") REFERENCES "DIM_BILLING_STATUS_JUNK" ("billing_status_key") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "DIM_ENERGY_CONTRACTS" ADD FOREIGN KEY ("company_id") REFERENCES "DIM_COMPANIES" ("company_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "DIM_COMPANIES" ADD FOREIGN KEY ("registered_office_region") REFERENCES "DIM_GEOGRAPHY" ("registered_office_region") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "DIM_COMPANIES" ADD FOREIGN KEY ("industry_sector") REFERENCES "DIM_SECTOR" ("industry_sector") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_FINANCIAL_STATEMENTS" ADD FOREIGN KEY ("company_id") REFERENCES "DIM_COMPANIES" ("company_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_INSOLVENCY_SNAPSHOT" ADD FOREIGN KEY ("company_id") REFERENCES "DIM_COMPANIES" ("company_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_INSOLVENCY_SNAPSHOT" ADD FOREIGN KEY ("snapshot_date") REFERENCES "DIM_DATE" ("date_day") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_CRM_TICKETS" ADD FOREIGN KEY ("created_at") REFERENCES "DIM_DATE" ("date_day") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_CRM_TICKETS" ADD FOREIGN KEY ("resolved_at") REFERENCES "DIM_DATE" ("date_day") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_CRM_TICKETS" ADD FOREIGN KEY ("company_id") REFERENCES "DIM_COMPANIES" ("company_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_CRM_TICKETS" ADD FOREIGN KEY ("ticket_category") REFERENCES "DIM_TICKET_CATEGORY" ("ticket_category") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_USER_LOGINS" ADD FOREIGN KEY ("login_timestamp") REFERENCES "DIM_DATE" ("date_day") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_USER_LOGINS" ADD FOREIGN KEY ("company_id") REFERENCES "DIM_COMPANIES" ("company_id") DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE "FACT_USER_LOGINS" ADD FOREIGN KEY ("device_type") REFERENCES "DIM_DEVICE" ("device_type") DEFERRABLE INITIALLY IMMEDIATE;
