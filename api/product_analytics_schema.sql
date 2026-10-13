-- Public, aggregate-only Product Analytics snapshots.
-- No customer-level records or identifiers are stored here.

create table if not exists public.product_analytics_metrics (
    id bigint generated always as identity primary key,
    total_customers bigint not null check (total_customers >= 0),
    total_events bigint not null check (total_events >= 0),
    total_sessions bigint not null check (total_sessions >= 0),
    total_logins bigint not null check (total_logins >= 0),
    total_product_views bigint not null check (total_product_views >= 0),
    avg_active_days numeric(12, 4) not null,
    avg_search_rate numeric(12, 6) not null,
    avg_cart_rate numeric(12, 6) not null,
    avg_purchase_rate numeric(12, 6) not null,
    avg_days_since_last_activity numeric(12, 4) not null,
    avg_days_since_last_login numeric(12, 4) not null,
    avg_days_since_last_product_view numeric(12, 4) not null,
    avg_events_per_active_day numeric(12, 4) not null,
    avg_sessions_per_active_day numeric(12, 4) not null,
    published_at timestamptz not null default now()
);

alter table public.product_analytics_metrics enable row level security;

grant select on public.product_analytics_metrics to anon, authenticated;

drop policy if exists "Public can read product analytics metrics"
    on public.product_analytics_metrics;

create policy "Public can read product analytics metrics"
on public.product_analytics_metrics
for select
to anon, authenticated
using (true);
