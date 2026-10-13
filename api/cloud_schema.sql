-- Cloud schema for the public Product Churn Analytics API.
-- Aggregate metrics only; no customer-level records.

create table if not exists public.churn_summary (
    id bigint generated always as identity primary key,
    total_customers bigint not null check (total_customers >= 0),
    churned_customers bigint not null check (churned_customers >= 0 and churned_customers <= total_customers),
    churn_rate numeric(7, 4) not null check (churn_rate >= 0 and churn_rate <= 1),
    published_at timestamptz not null default now()
);

create table if not exists public.churn_segments (
    id bigint generated always as identity primary key,
    segment_type text not null,
    segment_name text not null,
    total_customers bigint not null check (total_customers >= 0),
    churned_customers bigint not null check (churned_customers >= 0 and churned_customers <= total_customers),
    churn_rate numeric(7, 4) not null check (churn_rate >= 0 and churn_rate <= 1),
    published_at timestamptz not null default now(),
    unique (segment_type, segment_name, published_at)
);

create table if not exists public.model_metrics (
    id bigint generated always as identity primary key,
    model_name text not null,
    roc_auc numeric(6, 5) not null check (roc_auc >= 0 and roc_auc <= 1),
    average_precision numeric(6, 5) not null check (average_precision >= 0 and average_precision <= 1),
    evaluation_method text not null,
    published_at timestamptz not null default now()
);

-- Configure public read access separately after reviewing security policies.
-- Never expose a Supabase service-role key in website code.

-- Enable Row Level Security on all published metrics tables.
alter table public.churn_summary enable row level security;
alter table public.churn_segments enable row level security;
alter table public.model_metrics enable row level security;

-- Permit anonymous clients to read aggregate metrics only.
-- No public INSERT, UPDATE, or DELETE policies are created.
grant select on table public.churn_summary to anon, authenticated;
grant select on table public.churn_segments to anon, authenticated;
grant select on table public.model_metrics to anon, authenticated;

create policy "Public can read churn summary"
on public.churn_summary for select to anon, authenticated
using (true);

create policy "Public can read churn segments"
on public.churn_segments for select to anon, authenticated
using (true);

create policy "Public can read model metrics"
on public.model_metrics for select to anon, authenticated
using (true);
