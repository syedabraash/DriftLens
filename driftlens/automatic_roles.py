"""One action for cached camera repair, seed selection and role hypotheses."""
from __future__ import annotations

import json
from pathlib import Path

from .review import read_rows


def _saved_seed(summary: dict, frames: list[dict], observations: list[dict]):
    """Reuse an existing explicit seed without treating automatic maps as reviews."""
    continuation = summary.get('role_continuation') or {}
    seeds = continuation.get('seed_intervals', []) if continuation.get('active') else []
    origin = continuation.get('seed_origin', 'user_reviewed')
    if summary.get('role_review_status') == 'user_assignment_unverified':
        seeds = summary.get('role_intervals') or [{'start_clip_seconds':0,
            'end_clip_seconds':summary['duration_seconds'], 'lead_id':summary.get('lead_id'),
            'chase_id':summary.get('chase_id')}]
        origin = 'user_reviewed'
    by_frame = {}
    for observation in observations:
        if str(observation.get('observed',True)).lower() in {'true','1','1.0','yes'}:
            by_frame.setdefault(int(observation['frame_index']),set()).add(int(observation['track_id']))
    cuts = [float(current['clip_time']) for previous,current in zip(frames,frames[1:])
            if int(previous.get('shot_index',0)) != int(current.get('shot_index',0))]
    for saved in seeds:
        lead,chase = saved.get('lead_id'),saved.get('chase_id')
        if lead is None or chase is None or lead == chase:
            continue
        start = float(saved['start_clip_seconds'])
        end = min(float(saved['end_clip_seconds']),start+2,
                  next((cut for cut in cuts if cut>start),float(summary['duration_seconds'])))
        joint = [frame for frame in frames if start<=float(frame['clip_time'])<end
                 and {int(lead),int(chase)}.issubset(by_frame.get(int(frame['frame_index']),set()))]
        if len(joint) >= 3:
            return {'start_clip_seconds':start,'end_clip_seconds':end,
                    'lead_id':int(lead),'chase_id':int(chase),
                    'review_basis':'Reused the previously saved seed pair; later role assignments are automatic hypotheses.'},origin
    return None,None


def automatic_clip_roles(run_dir: Path) -> dict:
    """Continue a saved seed or propose one automatically; uncertainty stays visible."""
    from .role_continuity import follow_clip_roles
    from .run_analysis_ui import save_shot_analysis
    run_dir = Path(run_dir).resolve()
    original = json.loads((run_dir/'summary.json').read_text(encoding='utf-8-sig'))
    summary = original
    # Older uploads missed broadcast cuts. Publish a sibling, preserving inputs.
    if int(summary.get('visibility_revision',0))<4 and (run_dir/'candidates.csv').is_file():
        from .retrack import retrack_cached_run
        destination = run_dir.with_name(run_dir.name+'_auto')
        suffix = 2
        while destination.exists():
            destination = run_dir.with_name(f'{run_dir.name}_auto_{suffix}')
            suffix += 1
        summary = retrack_cached_run(run_dir,destination)
        run_dir = destination
    frames,observations = read_rows(run_dir/'frames.csv'),read_rows(run_dir/'observations.csv')
    seed,origin = _saved_seed(original,frames,observations)
    if seed is not None:
        proposal = {'status':'seed_ready','seed':seed,'pair_ids':[seed['lead_id'],seed['chase_id']],
                    'reason':'Using the previously saved starting pair; no repeated local ID entry is needed.',
                    'evidence':{'seed_origin':origin}}
    else:
        from .auto_seed import propose_automatic_seed
        proposal = propose_automatic_seed(Path(summary['source_path']),frames,observations,
                                           float(summary['duration_seconds']))
        seed,origin = proposal.get('seed'),'automatic_motion'
    if seed is None:
        summary['automatic_tandem'] = proposal
        revised = save_shot_analysis(run_dir,summary)
        return {'run_dir':run_dir,'summary':revised,'proposal':proposal,'status':proposal['status']}
    try:
        revised = follow_clip_roles(run_dir,seed,seed_origin=origin,automatic_proposal=proposal)
    except ValueError as error:
        # Insufficient distinct appearance evidence is an expected abstention.
        # Do not label it a successful match or delete previously saved roles.
        proposal = {**proposal,'status':'pair_only','seed':None,
                    'reason':f'Found a possible pair but could not match its appearance reliably: {error}'}
        summary['automatic_tandem'] = proposal
        revised = save_shot_analysis(run_dir,summary)
        return {'run_dir':run_dir,'summary':revised,'proposal':proposal,'status':'pair_only'}
    return {'run_dir':run_dir,'summary':revised,'proposal':proposal,'status':'matched'}


def render_automatic_roles(run_dir: Path, summary: dict) -> None:
    import streamlit as st
    st.markdown('**Automatic tandem matching**')
    st.caption('Use a saved starting pair or let the app look for two moving participants and their travel order. It then matches current cars across views. Unclear role order or ambiguous appearance stays unknown.')
    if st.button('Automatically find and follow tandem',type='primary',
                 key=f'automatic_tandem_{run_dir}',width='stretch'):
        try:
            with st.spinner('Checking camera views and matching the tandem…'):
                outcome = automatic_clip_roles(run_dir)
            st.session_state['pending_selected_run'] = str(outcome['run_dir'])
            st.query_params['run'] = outcome['run_dir'].name
            st.session_state['notice'] = ('Automatic tandem matching saved. Inspect MATCH labels and unknown spans.'
                if outcome['status']=='matched' else outcome['proposal']['reason'])
            st.rerun()
        except Exception as error:
            st.error(f'Automatic matching could not complete: {error}. Saved exports remain available. Choose one clear seed interval below if needed.')
    proposal = summary.get('automatic_tandem') or {}
    if proposal.get('status') in {'pair_only','unavailable'}:
        st.info(proposal.get('reason','The app could not determine a reliable starting pair.'))
        if proposal.get('pair_ids'):
            st.caption('Possible participant IDs: '+', '.join(map(str,proposal['pair_ids']))+'. Confirm lead and chase once in the seed controls below.')
