{% snapshot companies_snapshot %}

{{
    config(
      target_schema='snapshots',
      unique_key='id',
      strategy='check',
      check_cols=['legal_name', 'legal_form', 'industry_sector', 
                  'registered_office_region', 'is_active'],
    )
}}

select * from {{ source('db_source', 'companies') }}

{% endsnapshot %}
