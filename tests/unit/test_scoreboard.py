"""
Tests for the backtest scoreboard (story: "Backtest scoreboard").

- The backtest Lambda saves each analysis's scores (its headline metrics and the settings
  to filter on) as S3 metadata, with the Results tab's formulas (src/analysis_scores.py).
- The scoreboard lists everyone's saved analyses with those scores, without downloading them
  (src/saved_analyses.list_scored_analyses), and ranks and filters them (app/scoreboard.py).
- scripts/score_saved_analyses.py adds scores to analyses saved before the scoreboard.
- The Scoreboard tab opens an analysis in Results or adds it to the comparison.
"""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.compare import cagr, compact_analysis, outperformance, total_return
from app.scoreboard import (
    DEFAULT_RANK,
    METRICS,
    NOT_SCORED,
    TOP_ROWS,
    distinct_entries,
    filter_entries,
    filter_options,
    rank_entries,
    row_label,
    scoreboard_table,
)
from scripts.score_saved_analyses import score_saved_analyses
from src.analysis_scores import (
    MAX_SCORES_LENGTH,
    SCORES_FIELD,
    analysis_scores,
    decode_scores,
    encode_scores,
    score_metadata,
    score_result,
)
from src.backtest_jobs import decode_result, decode_saved, encode_saved
from src.backtest_service import BacktestService, result_file, summarize_job
from src.saved_analyses import FIRST_SAVED_FIELD, SavedAnalysisError, list_saved_analyses, list_scored_analyses
from tests.unit.test_backtest_lambda import BUCKET, PREFIX, _add, _job, _run, manager, s3  # noqa: F401 (fixtures)
from tests.unit.test_list_analyses import browser, new_session  # noqa: F401 (fixtures)
from tests.unit.test_saved_runs import _click_main_tab


def _saved(manager, analysis_id):
    """(job, result, metadata) of a saved result."""
    response = manager.client.get_object(Bucket=BUCKET, Key=f'{PREFIX}/{result_file(analysis_id)}')
    job, result = decode_saved(response['Body'].read())
    return job, decode_result(job['kind'], result), response['Metadata']


def _wrds_job_and_result(frequency='quarterly'):
    job = {'kind': 'backtest', 'data_mode': 'wrds', 'factors': ['roe', 'mom_12_1'],
           'factor_directions': {'roe': 'top', 'mom_12_1': 'bottom'},
           'factor_labels': {'roe': 'ROE', 'mom_12_1': '12-1 Momentum'}, 'frequency': frequency,
           'start_year': 2010, 'end_year': 2011, 'initial_aum': 5000.0, 'top_pct': 10.0,
           'use_market_cap_weight': True, 'factor_combination': 'integrating', 'restrict_fossil_fuels': False,
           'selected_sectors': [], 'cohort_pct': None, 'rebalance_date': None, 'trading_cost_bps': 25.0}
    returns = [2.0, -1.0, 3.0, 1.0]
    values = [5000.0]
    for r in returns:
        values.append(values[-1] * (1 + r / 100))
    dates = ['2010-03-31', '2010-06-30', '2010-09-30', '2010-12-31', '2011-03-31']
    result = {'final_value': values[-1], 'portfolio_values': values, 'yearly_returns': returns,
              'benchmark_returns': [1.0] * 4, 'years': dates, 'rebalance_dates': dates[:-1],
              'period_end_dates': dates[1:], 'frequency': frequency, 'periods_per_year': 4,
              'vol_raw_portfolio': 0.1, 'sharpe_portfolio': 0.9, 'max_drawdown_portfolio': -0.01,
              'win_rate': 0.75, 'information_ratio': float('nan')}
    return job, result


# ---------------------------------------------------------------------------
# Scores: the Results tab's metrics, saved with each analysis
# ---------------------------------------------------------------------------

def test_scores_match_the_compare_and_results_tabs(manager):
    analysis_id = BacktestService(manager).run(_job(factor_combination='integrating'))['hash']
    job, result, _ = _saved(manager, analysis_id)
    wrds_job, wrds_result = _wrds_job_and_result()
    for job, result in ((job, result), (wrds_job, wrds_result)):
        scores = score_result(job, result)
        compared = compact_analysis({'id': 'a' * 64, 'job': job, 'result': result, 'computed_at': None})
        assert scores['cagr'] == cagr(compared)
        assert scores['total_return'] == total_return(compared)
        assert scores['outperformance'] == outperformance(compared)
        assert scores['final_value'] == result['final_value']
        assert scores['sharpe'] == result['sharpe_portfolio']
        assert scores['max_drawdown'] == result['max_drawdown_portfolio']
        assert scores['volatility'] == result['vol_raw_portfolio']
        assert scores['win_rate'] == result['win_rate']
    assert score_result(wrds_job, {**wrds_result, 'sharpe_portfolio': float('nan')})['sharpe'] is None


def test_scores_have_the_settings_to_filter_on():
    job, result = _wrds_job_and_result()
    scores = analysis_scores(job, result)
    assert {k: scores[k] for k in ('mode', 'frequency', 'combination', 'start_year', 'end_year',
                                   'initial_aum', 'trading_cost_bps', 'factors')} == {
        'mode': 'wrds', 'frequency': 'Quarterly', 'combination': 'integrating', 'start_year': 2010,
        'end_year': 2011, 'initial_aum': 5000.0, 'trading_cost_bps': 25.0, 'factors': ['ROE', '12-1 Momentum']}
    legacy = analysis_scores(_job(), {'final_value': 1100.0, 'yearly_returns': [10.0]})
    assert (legacy['mode'], legacy['frequency'], legacy['trading_cost_bps']) == ('legacy', 'Annual', 0.0)


def test_scores_are_short_ascii_json_and_read_back():
    job, result = _wrds_job_and_result()
    text = score_metadata(job, result)[SCORES_FIELD]
    assert text.isascii() and len(text) <= MAX_SCORES_LENGTH
    assert decode_scores(text) == json.loads(json.dumps(analysis_scores(job, result)))
    assert decode_scores(text)['metrics']['sharpe'] == 0.9
    # Long factor labels are dropped (the summary still names them), never cut mid-label
    long_job = {**job, 'factor_labels': {'roe': 'é' * 900, 'mom_12_1': 'x' * 900}}
    long_text = encode_scores(analysis_scores(long_job, result))
    assert len(long_text) <= MAX_SCORES_LENGTH and decode_scores(long_text)['factors'] is None
    for bad in (None, '', 'not json', '[]', '{"v": 99, "metrics": {}}', '{"v": 1}'):
        assert decode_scores(bad) is None
    assert score_metadata({**job, 'kind': 'cohort'}, result) == {}


def test_the_lambda_saves_scores_with_each_analysis_only(manager):
    service = BacktestService(manager)
    analysis_id = service.run(_job())['hash']
    cohort_id = service.run(_job('cohort', cohort_pct=20))['hash']
    job, result, metadata = _saved(manager, analysis_id)
    assert decode_scores(metadata[SCORES_FIELD]) == json.loads(encode_scores(analysis_scores(job, result)))
    assert sum(len(k) + len(v) for k, v in metadata.items()) < 2048           # S3's metadata limit
    assert SCORES_FIELD not in _saved(manager, cohort_id)[2]


# ---------------------------------------------------------------------------
# Listing everyone's scores from S3
# ---------------------------------------------------------------------------

def test_scored_listing_has_every_analysis_newest_first(manager, s3):
    service = BacktestService(manager)
    first = service.run(_job())['hash']
    service.run(_job('cohort', cohort_pct=20))                                # not an analysis
    second = service.run(_job(factor_combination='integrating'))['hash']
    old = 'c' * 64
    s3.put_object(BUCKET, f'{PREFIX}/{result_file(old)}', b'{}')             # saved before scores
    listing = list_scored_analyses(manager)
    assert [e['analysis_id'] for e in listing['entries']] == [old, second, first]
    assert listing['total'] == 4
    assert listing['entries'][0]['scores'] is None
    assert listing['entries'][1]['scores']['combination'] == 'integrating'
    assert listing['entries'][1]['summary'] == summarize_job(_job(factor_combination='integrating'))


def test_listings_keep_the_first_saved_time_of_rewritten_metadata(manager, s3):
    service = BacktestService(manager)
    first = service.run(_job())['hash']
    second = service.run(_job(initial_aum=2000))['hash']
    key = f'{PREFIX}/{result_file(first)}'
    body, metadata = s3.objects[key]
    original = s3.modified[key]
    s3.copy_object(BUCKET, key, {'Bucket': BUCKET, 'Key': key},
                   {**metadata, FIRST_SAVED_FIELD: original.isoformat()}, 'REPLACE')   # now the newest object
    for listing in (list_scored_analyses(manager), list_saved_analyses(manager)):
        assert [e['analysis_id'] for e in listing['entries']] == [second, first]
        assert listing['entries'][1]['saved_at'] == original


def test_scored_listing_without_permission_is_explained(manager, s3):
    s3.can_list = False
    with pytest.raises(SavedAnalysisError, match='may not list saved analyses'):
        list_scored_analyses(manager)


# ---------------------------------------------------------------------------
# Scoring analyses saved before the scoreboard (scripts/score_saved_analyses.py)
# ---------------------------------------------------------------------------

def test_backfill_scores_old_analyses_in_place(manager, s3):
    service = BacktestService(manager)
    analysis_id = service.run(_job())['hash']
    cohort_id = service.run(_job('cohort', cohort_pct=20))['hash']
    key, cohort_key = (f'{PREFIX}/{result_file(i)}' for i in (analysis_id, cohort_id))
    scored_metadata = dict(s3.objects[key][1])
    body = s3.objects[key][0]
    s3.objects[key] = (body, {'computed_at': '2026-10-01T00:00:00Z'})        # saved before summaries and scores
    s3.objects[cohort_key] = (s3.objects[cohort_key][0], {})
    pre_share = 'd' * 64
    s3.put_object(BUCKET, f'{PREFIX}/{result_file(pre_share)}', json.dumps({'final_value': 1.0}).encode())
    saved_at = {i: s3.modified[f'{PREFIX}/{result_file(i)}'] for i in (analysis_id, cohort_id, pre_share)}
    order = [e['analysis_id'] for e in list_scored_analyses(manager)['entries']]

    puts = len(s3.puts)
    counts = score_saved_analyses(manager, apply=False, log=lambda _: None)
    assert (counts['scored'], counts['labeled'], counts['no job']) == (1, 1, 1)
    assert len(s3.puts) == puts                                              # a dry run writes nothing

    score_saved_analyses(manager, apply=True, log=lambda _: None)
    assert s3.objects[key][0] == body                                        # the analysis is unchanged
    metadata = s3.objects[key][1]
    assert metadata[SCORES_FIELD] == scored_metadata[SCORES_FIELD]           # as the Lambda saves them
    assert metadata['summary'] == scored_metadata['summary'] and metadata['kind'] == 'backtest'
    assert metadata['computed_at'] == '2026-10-01T00:00:00Z'
    assert s3.objects[cohort_key][1]['kind'] == 'cohort' and SCORES_FIELD not in s3.objects[cohort_key][1]
    listing = list_scored_analyses(manager)
    assert [e['analysis_id'] for e in listing['entries']] == [i for i in order if i != cohort_id]
    assert {e['analysis_id']: e['saved_at'] for e in listing['entries']} == {
        i: saved_at[i] for i in (analysis_id, pre_share)}
    assert score_saved_analyses(manager, apply=True, log=lambda _: None)['already scored'] == 1   # once only


# ---------------------------------------------------------------------------
# Ranking and filtering (app/scoreboard.py)
# ---------------------------------------------------------------------------

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _entry(n, sharpe=None, volatility=0.2, mode='legacy', frequency='Annual', combination='mixing',
           years=(2002, 2024), factors=('ROE',), scored=True, minutes=0):
    scores = None
    if scored:
        scores = {'v': 1, 'mode': mode, 'frequency': frequency, 'combination': combination,
                  'start_year': years[0], 'end_year': years[1], 'initial_aum': 1000.0, 'trading_cost_bps': 0.0,
                  'factors': list(factors),
                  'metrics': {'sharpe': sharpe, 'volatility': volatility, 'cagr': 0.1, 'outperformance': 0.05,
                              'max_drawdown': -0.3, 'win_rate': 0.6, 'total_return': 2.0, 'final_value': 3000.0}}
    return {'analysis_id': f'{n:x}' * 64, 'saved_at': T0 + timedelta(minutes=minutes),
            'summary': f'Analysis {n}', 'scores': scores}


def test_ranking_is_best_first_with_shared_ranks_for_ties():
    entries = [_entry(1, sharpe=0.5, minutes=1), _entry(2, sharpe=0.9), _entry(3, sharpe=0.50001, minutes=5),
               _entry(4, scored=False, minutes=9), _entry(5, sharpe=None, minutes=8), _entry(6, sharpe=0.1)]
    ranked = rank_entries(entries)
    assert DEFAULT_RANK == 'Sharpe'
    # 0.5 and 0.50001 look the same at 4 decimals: they tie, newest first, and share rank 2
    assert [(e['summary'], e['rank']) for e in ranked] == [
        ('Analysis 2', 1), ('Analysis 3', 2), ('Analysis 1', 2), ('Analysis 6', 4),
        ('Analysis 5', None), ('Analysis 4', None)]                           # n/a, then not scored
    assert [e['summary'] for e in rank_entries(entries[:3], 'Volatility')] == ['Analysis 3', 'Analysis 1',
                                                                              'Analysis 2']
    assert row_label(ranked[0]) == f"#1 · Analysis 2 · {'2' * 8}"
    assert row_label(ranked[-1]).startswith(f"{NOT_SCORED} · Analysis 4")


def test_runs_of_one_strategy_are_listed_once_as_the_newest():
    entries = [_entry(1, sharpe=0.5, minutes=1), _entry(2, sharpe=0.5, minutes=3), _entry(3, sharpe=0.50001, minutes=2),
               _entry(4, sharpe=0.6, minutes=4), _entry(5, sharpe=0.5, minutes=5, combination='integrating')]
    entries[1]['scores']['initial_aum'] = 1008824.03                         # another amount: same returns
    entries[1]['scores']['metrics']['final_value'] = 3_026_472.09
    kept, folded = distinct_entries(entries)
    # 1, 2 and 3 are one strategy with the same rates as shown; 4 has other returns, 5 other settings
    assert [e['analysis_id'][0] for e in kept] == ['2', '4', '5'] and folded == 2


def test_lower_volatility_ranks_first():
    ranked = rank_entries([_entry(1, volatility=0.3), _entry(2, volatility=0.1)], 'Volatility')
    assert [e['summary'] for e in ranked] == ['Analysis 2', 'Analysis 1']


def test_filters_by_source_combination_factors_and_period():
    entries = [_entry(1, mode='wrds', frequency='Quarterly', factors=('ROE', 'ROA'), years=(2010, 2020)),
               _entry(2, combination='integrating', factors=('ROE',)),
               _entry(3, factors=('ROA',), years=(2002, 2010)),
               _entry(4, scored=False)]
    options = filter_options(entries)
    assert options == {'sources': ['Legacy, Annual', 'WRDS, Quarterly'], 'combinations': ['Mixing', 'Integrating'],
                       'factors': ['ROA', 'ROE'], 'years': (2002, 2024)}

    def ids(**filters):
        return [e['summary'][-1] for e in filter_entries(entries, **filters)]
    assert ids() == ['1', '2', '3', '4']                                      # no filter: everything
    assert ids(sources=['WRDS, Quarterly']) == ['1']
    assert ids(combinations=['Integrating']) == ['2']
    assert ids(factors=['ROE']) == ['1', '2']
    assert ids(factors=['ROE', 'ROA']) == ['1']                               # every chosen factor
    assert ids(years=(2002, 2015)) == ['3']                                   # the whole period within
    assert ids(sources=['Legacy, Annual'], combinations=['Mixing']) == ['3']


def test_table_shows_percents_and_blanks_for_unscored():
    table = scoreboard_table(rank_entries([_entry(1, sharpe=0.8), _entry(2, scored=False)]))
    assert list(table.columns) == ['Rank', 'Analysis', 'Source', 'Period', *[m.name for m in METRICS],
                                   'Saved (UTC)', 'ID']
    first, unscored = table.iloc[0], table.iloc[1]
    assert (first['Rank'], first['Source'], first['Period']) == (1, 'Legacy, Annual', '2002-2024')
    assert first['CAGR'] == pytest.approx(10.0) and first['Sharpe'] == 0.8 and first['Final Value'] == 3000.0
    assert first['Saved (UTC)'] == '2026-11-01 00:00' and first['ID'] == '1' * 8
    assert unscored['Source'] == NOT_SCORED and pd.isna(unscored['Rank']) and pd.isna(unscored['Sharpe'])


# ---------------------------------------------------------------------------
# The Scoreboard tab
# ---------------------------------------------------------------------------

def _scoreboard(at):
    _click_main_tab(at, 'Scoreboard')
    return _table(at)


def _table(at):
    """The scoreboard's table (the tab's only dataframe), or None when it shows none."""
    return at.dataframe[0].value if len(at.dataframe) else None


def test_scoreboard_ranks_everyones_analyses_and_opens_one(new_session, manager):
    colleague = BacktestService(manager).run(_job(initial_aum=5000))['hash']   # someone else's run
    at = new_session()
    _add(at, 'ROE using 9/30 Data')
    mine = _run(at)['compute']['hash']

    table = _scoreboard(at)
    assert sorted(table['ID']) == sorted([colleague[:8], mine[:8]])
    sharpe = {analysis_id[:8]: _saved(manager, analysis_id)[1]['sharpe_portfolio'] for analysis_id in (colleague, mine)}
    assert list(table['ID']) == sorted(sharpe, key=sharpe.get, reverse=True)  # best Sharpe first
    assert list(table['Rank']) == [1, 2]
    assert table.loc[table['ID'] == colleague[:8], 'Sharpe'].iloc[0] == sharpe[colleague[:8]]

    at.selectbox(key='scoreboard_rank_by').set_value('Volatility').run()
    volatility = {i[:8]: _saved(manager, i)[1]['vol_raw_portfolio'] for i in (colleague, mine)}
    assert list(_table(at)['ID']) == sorted(volatility, key=volatility.get)

    at.selectbox(key='scoreboard_pick').set_value(colleague).run()
    assert colleague in [c.value for c in at.code]
    at.button(key='scoreboard_open').click().run()
    assert at.session_state['main_tabs'] == 'Results'
    opened = at.session_state['saved_runs'][0]
    assert opened['settings']['initial_aum'] == 5000
    assert any('from the scoreboard' in s.value for s in at.success)

    _click_main_tab(at, 'Scoreboard')                                         # the rank stays as chosen
    assert at.selectbox(key='scoreboard_rank_by').value == 'Volatility'


def test_scoreboard_adds_an_analysis_to_the_comparison(new_session, manager):
    analysis_id = BacktestService(manager).run(_job())['hash']
    at = new_session()
    _scoreboard(at)
    at.button(key='scoreboard_compare').click().run()
    assert at.session_state['main_tabs'] == 'Compare'
    assert analysis_id in at.session_state['compare_ids'].values()


def test_scoreboard_filters_and_leaves_out_unscored(new_session, manager, s3):
    service = BacktestService(manager)
    mixing = service.run(_job())['hash']
    integrating = service.run(_job(factor_combination='integrating'))['hash']
    old = 'e' * 64
    s3.put_object(BUCKET, f'{PREFIX}/{result_file(old)}', encode_saved(_job(), {'final_value': 1.0}))
    at = new_session()
    table = _scoreboard(at)
    assert sorted(table['ID']) == sorted([mixing[:8], integrating[:8]])                # old is not ranked
    assert any('1 analysis saved before the scoreboard is not ranked' in c.value for c in at.caption)
    at.multiselect(key='scoreboard_combinations').set_value(['Integrating']).run()
    assert list(_table(at)['ID']) == [integrating[:8]]


def test_scoreboard_shows_the_top_rows_until_show_all(new_session, manager, s3):
    for n in range(TOP_ROWS + 2):
        job, result = _wrds_job_and_result()
        job = {**job, 'initial_aum': 1000.0 + n}
        s3.put_object(BUCKET, f'{PREFIX}/{result_file(f"{n:064x}")}', encode_saved(job, result),
                      Metadata={'kind': 'backtest', 'summary': f'Run {n}', **score_metadata(job, result)})
    at = new_session()
    assert len(_scoreboard(at)) == TOP_ROWS
    at.toggle(key='scoreboard_show_all').set_value(True).run()
    assert len(_table(at)) == TOP_ROWS + 2


def test_scoreboard_explains_a_listing_error_and_an_empty_bucket(new_session, s3):
    at = new_session()
    assert _scoreboard(at) is None
    assert any('No scored analyses yet' in i.value for i in at.info)
    s3.can_list = False
    import streamlit as st
    st.cache_data.clear()
    at.run()
    assert any('may not list saved analyses' in e.value for e in at.error)
