import re
from collections import deque
import numpy as np
import pandas as pd


def build_weekly_entry_history(con, source_view='weekly_trades', output_table='week_history'):
    """Create history at every observed wallet/time. Identifiers must be simple names.

    Joining at a target market's first supplied-week timestamp implicitly excludes
    that target market: it has no earlier supplied-week rows. This does not hold
    for arbitrary later prediction times. Same-time rows across markets are all
    excluded. HHI includes positive cumulative BUY dollars; sell-only markets
    still count toward observed market breadth.
    """
    import re
    for name in (source_view, output_table):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError('DuckDB identifiers must contain only letters, digits and underscores')
    # Materialize compact intermediate batches to avoid repeated raw scans.
    con.execute(f'''CREATE OR REPLACE TEMP TABLE _wh_market_batches AS
        SELECT wallet,market,time, count(*)::BIGINT AS n,
               sum(CASE WHEN side='BUY' THEN dollars ELSE 0 END)::DOUBLE AS b
        FROM {source_view} GROUP BY wallet,market,time''')
    con.execute('''CREATE OR REPLACE TEMP TABLE _wh_market_cumulative AS
        SELECT *, sum(b) OVER (PARTITION BY wallet,market ORDER BY time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_b,
               row_number() OVER (PARTITION BY wallet,market ORDER BY time) AS market_batch
        FROM _wh_market_batches''')
    con.execute('''CREATE OR REPLACE TEMP TABLE _wh_wallet_batches AS
        SELECT wallet,time,sum(n)::BIGINT AS n,sum(b) AS b,
               sum(b*(2*cum_b-b)) AS square_delta,
               sum(CASE WHEN market_batch=1 THEN 1 ELSE 0 END)::BIGINT AS new_markets
        FROM _wh_market_cumulative GROUP BY wallet,time''')
    con.execute(f'''CREATE OR REPLACE TABLE {output_table} AS
        WITH prior AS (
            SELECT wallet,time,
                coalesce(sum(n) OVER w,0)::BIGINT AS rows_before,
                count(*) OVER w AS timestamps_before,
                coalesce(sum(new_markets) OVER w,0)::BIGINT AS markets_before,
                coalesce(sum(b) OVER w,0)::DOUBLE AS buys_before,
                coalesce(sum(square_delta) OVER w,0)::DOUBLE AS square_before,
                min(time) OVER w AS first_before, max(time) OVER w AS latest_before
            FROM _wh_wallet_batches
            WINDOW w AS (PARTITION BY wallet ORDER BY time
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ), metrics AS (
            SELECT *, CASE WHEN buys_before>0 THEN square_before/(buys_before*buys_before)
                           ELSE NULL END AS hhi,
                epoch(time-first_before)/3600.0 AS span_hours,
                epoch(time-latest_before)/3600.0 AS recency_hours
            FROM prior
        )
        SELECT wallet,time,
            rows_before AS feature_week_history_fill_rows,
            timestamps_before AS feature_week_history_distinct_timestamps,
            markets_before AS feature_week_history_markets,
            buys_before AS feature_week_history_buy_dollars,
            hhi AS feature_week_history_buy_market_hhi,
            first_before AS feature_week_history_first_time,
            latest_before AS feature_week_history_latest_time,
            span_hours AS feature_week_history_observed_span_hours,
            recency_hours AS feature_week_history_recency_hours,
            ln(1+recency_hours) AS feature_week_history_log_recency_hours,
            ln(1+rows_before) AS feature_week_history_log_fill_rows,
            ln(1+timestamps_before) AS feature_week_history_log_distinct_timestamps,
            ln(1+markets_before) AS feature_week_history_log_markets,
            ln(1+buys_before) AS feature_week_history_log_buy_dollars,
            (rows_before=0)::INTEGER AS feature_week_history_no_observed_rows,
            (buys_before=0)::INTEGER AS feature_week_history_no_buy,
            0::INTEGER AS feature_week_history_prior_conditional_profit_history_available
        FROM metrics''')
    for table in ('_wh_market_batches', '_wh_market_cumulative', '_wh_wallet_batches'):
        con.execute(f'DROP TABLE {table}')
    return output_table

def stream_fixed_histories(records,entries,week_start_ms,window_ms=86400000,float_tolerance=1e-8):
    from collections import deque
    import math
    output=[];max_square_drift=0.;last_wallet=None;last_time=None
    for wallet,market,t,n,nb,b in records:
        t=int(t);n=int(n);nb=int(nb);b=float(b)
        if wallet!=last_wallet:
            queue=deque();market_rows={};market_buy={};market_buy_rows={};time_counts={}
            total_rows=buy_rows=0;buy_dollars=0.;square=0.
            cumulative_rows=cumulative_buy_rows=cumulative_timestamps=0;cumulative_dollars=0.;cumulative_markets=set()
            last_wallet=wallet;last_time=None
        if t!=last_time:
            # Lower endpoint INCLUDED: remove only records strictly older than T-24h.
            while queue and queue[0][0]<t-window_ms:
                ot,om,on,onb,ob=queue.popleft()
                total_rows-=on;buy_rows-=onb;buy_dollars-=ob
                old=market_buy.get(om,0.);new=old-ob
                square+=new*new-old*old
                market_rows[om]-=on;market_buy_rows[om]-=onb
                if market_buy_rows[om]==0:new=0.
                if market_rows[om]==0:
                    market_rows.pop(om);market_buy.pop(om,None);market_buy_rows.pop(om,None)
                else:
                    tolerance=float_tolerance*max(1.,abs(old),abs(ob))
                    assert new>=-tolerance,('negative_buy_sum',wallet,om,new)
                    market_buy[om]=0. if -tolerance<=new<0 else new
                time_counts[ot]-=1
                if time_counts[ot]==0:time_counts.pop(ot)
            if total_rows==0:
                # Exact empty state, not a zero-fill assumption about missing pre-week history.
                buy_dollars=0.;square=0.;assert buy_rows==0 and not market_rows and not time_counts
            elif buy_dollars<0:
                assert abs(buy_dollars)<=float_tolerance*max(1.,cumulative_dollars)
                buy_dollars=0.
            ids=entries.get(wallet,{}).get(t,[])
            if ids:
                # Recompute HHI numerator at every emitted cutoff to prevent subtraction drift.
                exact_square=sum(value*value for value in market_buy.values())
                exact_dollars=sum(market_buy.values())
                assert abs(exact_dollars-buy_dollars)<=float_tolerance*max(1.,cumulative_dollars)
                buy_dollars=exact_dollars
                max_square_drift=max(max_square_drift,abs(exact_square-square)/max(1.,exact_square))
                assert abs(exact_square-square)<=float_tolerance*max(1.,exact_square,cumulative_dollars*cumulative_dollars)
                square=exact_square
                hhi=exact_square/(buy_dollars*buy_dollars) if buy_dollars>0 else math.nan
                if buy_dollars>0:assert -float_tolerance<=hhi<=1+float_tolerance,(wallet,t,hhi)
                full=t-window_ms>=week_start_ms
                row={'full24h_observed_span':int(full),'week_buy_fill_rows':cumulative_buy_rows,
                     'week_avg_buy_ticket':cumulative_dollars/cumulative_buy_rows if cumulative_buy_rows else math.nan,
                     'proof_week_fill_rows':cumulative_rows,'proof_week_markets':len(cumulative_markets),
                     'proof_week_buy_dollars':cumulative_dollars,'proof_week_distinct_timestamps':cumulative_timestamps}
                stats={'fill_rows':total_rows,'distinct_timestamps':len(time_counts),'markets':len(market_rows),
                    'buy_fill_rows':buy_rows,'buy_dollars':buy_dollars,'buy_market_hhi':hhi,
                    'avg_buy_ticket':buy_dollars/buy_rows if buy_rows else math.nan,
                    'no_observed_rows':int(total_rows==0),'no_buy_rows':int(buy_rows==0),'no_buy_dollars':int(buy_dollars==0)}
                for key,value in stats.items():row['fixed24h_'+key]=value if full else math.nan
                for pair_id in ids:output.append({'pair_id':pair_id,**row})
            # Emit BEFORE any rows in the entire T batch, even across markets.
            cumulative_timestamps+=1;last_time=t
        old=market_buy.get(market,0.);new=old+b;square+=new*new-old*old
        market_buy[market]=new;market_rows[market]=market_rows.get(market,0)+n;market_buy_rows[market]=market_buy_rows.get(market,0)+nb
        time_counts[t]=time_counts.get(t,0)+1
        queue.append((t,market,n,nb,b));total_rows+=n;buy_rows+=nb;buy_dollars+=b
        cumulative_rows+=n;cumulative_buy_rows+=nb;cumulative_dollars+=b;cumulative_markets.add(market)
    return output,{'max_relative_maintained_square_drift':max_square_drift,'emitted_rows':len(output),
                   'float_tolerance':float_tolerance,'HHI_numerator_recomputed_at_every_emitted_cutoff':True}

def fixed_history_fixture():
    import math
    day=86400000
    # At T=2days: lower boundary1day included, just-before boundary excluded, allT ties excluded.
    records=[('w','old',day-1,1,1,9.),('w','a',day,1,1,2.),
             ('w','b',day+1,1,0,0.),('w','target',2*day,1,1,99.),
             ('w','tied_other',2*day,1,1,50.),('z','zero',day,1,1,0.),('z','target',2*day,1,1,5.)]
    entries={'w':{2*day:[0]},'z':{2*day:[1]}}
    rows,_=stream_fixed_histories(records,entries,0)
    first,zero=rows
    assert first['fixed24h_fill_rows']==2 and first['fixed24h_markets']==2
    assert first['fixed24h_buy_dollars']==2 and first['fixed24h_buy_market_hhi']==1
    assert first['fixed24h_buy_fill_rows']==1 and first['fixed24h_avg_buy_ticket']==2
    assert zero['fixed24h_buy_fill_rows']==1 and zero['fixed24h_avg_buy_ticket']==0
    assert math.isnan(zero['fixed24h_buy_market_hhi']) and zero['fixed24h_no_buy_rows']==0 and zero['fixed24h_no_buy_dollars']==1
    early,_=stream_fixed_histories([('w','target',1,1,1,1.)],{'w':{1:[0]}},0)
    assert early[0]['full24h_observed_span']==0 and math.isnan(early[0]['fixed24h_fill_rows'])
    empty,_=stream_fixed_histories([('w','target',day,1,1,1.)],{'w':{day:[0]}},0)
    assert empty[0]['fixed24h_fill_rows']==0 and math.isnan(empty[0]['fixed24h_avg_buy_ticket'])
    print('Strict24h fixture passed: lower inclusion, upper ties, target exclusion, sell breadth, zero-dollar BUY, truncation and empty history.')

def event_proxy(slug):
    weather=re.match(r'((?:highest|lowest)-temperature-in-.+?-on-[a-z]+-\d+-\d{4})(?:-|$)',slug)
    if weather:return weather[1],'weather_city_date_proxy'
    dated=re.match(r'(.+?-\d{4}-\d{2}-\d{2})(?:-|$)',slug)
    if dated:return dated[1],'dated_match_prefix_proxy'
    crypto=re.match(r'([a-z]+)-updown-(?:5m|15m|1h|4h)-(\d+)$',slug)
    if crypto:return 'crypto-day-'+str(int(crypto[2])//86400),'all_asset_crypto_UTC_day_proxy'
    threshold=re.match(r'(.+?)-above-\d+(-on-.+)$',slug)
    if threshold:return threshold[1]+threshold[2],'asset_time_threshold_proxy'
    return slug,'unmapped_market_as_proxy'