
import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="RunRec MVP",
    page_icon="R",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

GLOBAL_RANKING_PATH = ARTIFACTS_DIR / "global_ranking.json"
PLAN_PRIOR_PATH = ARTIFACTS_DIR / "plan_action_prior.json"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
SAFETY_AUDIT_PATH = ARTIFACTS_DIR / "safety_audit.json"
SAFETY_RULES_PATH = ARTIFACTS_DIR / "safety_rules.json"

TOP_K = 3
TSB_THRESHOLD = -20
ATL_SPAN = 2
CTL_SPAN = 6

ACTION_SPACE = [
    "easy__very_short",
    "easy__short",
    "easy__medium",
    "easy__long",
    "tempo__short",
    "tempo__medium",
    "intervals__short",
    "intervals__medium",
    "long__long",
    "long__very_long",
]

REST_ACTION = "rest__none"

WORKOUT_TYPE_LABELS = {
    "easy": "Treino fácil",
    "tempo": "Treino tempo",
    "intervals": "Treino intervalado",
    "long": "Longão",
    "rest": "Descanso",
}

DISTANCE_BUCKET_LABELS = {
    "very_short": "muito curto",
    "short": "curto",
    "medium": "médio",
    "long": "longo",
    "very_long": "muito longo",
    "none": "sem distância",
}

DISTANCE_BUCKET_RANGES = {
    "very_short": "até 5 km",
    "short": "5 a 10 km",
    "medium": "10 a 16 km",
    "long": "16 a 24 km",
    "very_long": "24 km ou mais",
    "none": "0 km",
}

DISTANCE_BUCKET_ORDER = [
    "very_short",
    "short",
    "medium",
    "long",
    "very_long",
]

INTENSITY_FACTORS = {
    "Predominantemente leve": 0.50,
    "Misto / moderado": 0.60,
    "Com bastante intensidade": 0.75,
}

OBJECTIVE_PROFILES = {
    "Ganhar consistência": {
        "easy": 0.42,
        "long": 0.24,
        "tempo": 0.20,
        "intervals": 0.14,
    },
    "Melhorar velocidade": {
        "tempo": 0.34,
        "intervals": 0.32,
        "easy": 0.28,
        "long": 0.06,
    },
    "Construir endurance": {
        "long": 0.36,
        "easy": 0.32,
        "tempo": 0.18,
        "intervals": 0.14,
    },
}

LAST_WORKOUT_LABELS = {
    "Descanso": "rest",
    "Treino fácil": "easy",
    "Treino tempo": "tempo",
    "Treino intervalado": "intervals",
    "Longão": "long",
}


@st.cache_data
def load_json_if_exists(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def deduplicate_preserving_order(values):
    deduplicated = []

    for value in values:
        if value not in deduplicated:
            deduplicated.append(value)

    return deduplicated


def load_global_ranking():
    ranking = load_json_if_exists(GLOBAL_RANKING_PATH)

    if not ranking:
        return ACTION_SPACE

    return deduplicate_preserving_order(ranking + ACTION_SPACE)


def load_plan_prior():
    prior = load_json_if_exists(PLAN_PRIOR_PATH)

    if not prior:
        return {}

    return prior


def action_workout_type(action_id):
    return action_id.split("__")[0]


def action_distance_bucket(action_id):
    return action_id.split("__")[1]


def action_to_label(action_id):
    workout_type = action_workout_type(action_id)
    distance_bucket = action_distance_bucket(action_id)

    workout_label = WORKOUT_TYPE_LABELS.get(workout_type, workout_type)
    distance_label = DISTANCE_BUCKET_LABELS.get(distance_bucket, distance_bucket)

    return f"{workout_label} {distance_label}"


def action_to_output(action_id):
    workout_type = action_workout_type(action_id)
    distance_bucket = action_distance_bucket(action_id)

    return {
        "tipo_treino": WORKOUT_TYPE_LABELS.get(workout_type, workout_type),
        "distancia_recomendada": DISTANCE_BUCKET_RANGES.get(
            distance_bucket,
            distance_bucket,
        ),
        "faixa_distancia": DISTANCE_BUCKET_LABELS.get(
            distance_bucket,
            distance_bucket,
        ),
        "action_id": action_id,
    }


def assign_fatigue_state(tsb):
    if tsb <= TSB_THRESHOLD:
        return "high_fatigue"
    if tsb >= abs(TSB_THRESHOLD):
        return "fresh"
    return "balanced"


def assign_recovery_state(days_since_intense):
    if days_since_intense < 2:
        return "recent_intensity"
    if days_since_intense <= 5:
        return "moderate_recovery"
    return "recovered"


def classify_readiness_state(
    avg_weekly_km,
    last_weekly_km,
    max_recent_weekly_km,
    comfortable_pace,
    current_status,
    weeks_without_running,
):
    if current_status == "Não estou correndo / só caminhada":
        return "return_to_activity"

    if max_recent_weekly_km <= 5 and (
        comfortable_pace >= 9 or weeks_without_running >= 6
    ):
        return "return_to_activity"

    if current_status == "Retomando com caminhada + trote":
        return "base_building"

    if weeks_without_running >= 4 and avg_weekly_km < 20:
        return "base_building"

    if avg_weekly_km < 12 or last_weekly_km < 6:
        return "base_building"

    if comfortable_pace >= 8.5 and avg_weekly_km < 25:
        return "base_building"

    return "active_runner"


def build_user_state(
    weeks_df,
    comfortable_pace,
    days_since_intense,
    current_status,
    weeks_without_running,
    last_workout_type,
):
    state_df = weeks_df.copy()

    state_df["weekly_km"] = pd.to_numeric(
        state_df["weekly_km"],
        errors="coerce",
    ).fillna(0)

    state_df["intensity_factor"] = state_df["intensity_profile"].map(
        INTENSITY_FACTORS,
    )

    state_df["weekly_trimp"] = (
        state_df["weekly_km"]
        * comfortable_pace
        * state_df["intensity_factor"]
    )

    state_df["atl"] = state_df["weekly_trimp"].ewm(
        span=ATL_SPAN,
        adjust=False,
    ).mean()

    state_df["ctl"] = state_df["weekly_trimp"].ewm(
        span=CTL_SPAN,
        adjust=False,
    ).mean()

    state_df["tsb"] = state_df["ctl"] - state_df["atl"]

    last_row = state_df.iloc[-1]
    avg_weekly_km = state_df["weekly_km"].mean()
    last_weekly_km = float(last_row["weekly_km"])
    max_recent_weekly_km = float(state_df["weekly_km"].tail(2).max())
    prev_tsb = float(last_row["tsb"])

    readiness_state = classify_readiness_state(
        avg_weekly_km=avg_weekly_km,
        last_weekly_km=last_weekly_km,
        max_recent_weekly_km=max_recent_weekly_km,
        comfortable_pace=comfortable_pace,
        current_status=current_status,
        weeks_without_running=weeks_without_running,
    )

    return {
        "prev_weekly_km": last_weekly_km,
        "avg_weekly_km": float(avg_weekly_km),
        "max_recent_weekly_km": max_recent_weekly_km,
        "prev_atl": float(last_row["atl"]),
        "prev_ctl": float(last_row["ctl"]),
        "prev_tsb": prev_tsb,
        "days_since_intense": float(days_since_intense),
        "prev_is_inactive_week": bool(last_row["weekly_km"] == 0),
        "current_status": current_status,
        "weeks_without_running": int(weeks_without_running),
        "last_workout_type": last_workout_type,
        "readiness_state": readiness_state,
        "fatigue_state": assign_fatigue_state(prev_tsb),
        "recovery_state": assign_recovery_state(float(days_since_intense)),
        "state_df": state_df,
    }


def target_bucket_for_action(workout_type, user_state):
    avg_weekly_km = user_state["avg_weekly_km"]
    readiness_state = user_state["readiness_state"]

    if readiness_state == "return_to_activity":
        return "very_short"

    if readiness_state == "base_building":
        if avg_weekly_km < 10:
            return "very_short"
        return "short"

    if workout_type == "intervals":
        if avg_weekly_km < 35:
            return "short"
        return "medium"

    if workout_type == "tempo":
        if avg_weekly_km < 25:
            return "short"
        return "medium"

    if workout_type == "long":
        if avg_weekly_km < 28:
            return "medium"
        if avg_weekly_km < 55:
            return "long"
        return "very_long"

    if avg_weekly_km < 18:
        return "short"
    if avg_weekly_km < 40:
        return "medium"
    return "long"


def distance_fit_score(workout_type, distance_bucket, user_state):
    if distance_bucket not in DISTANCE_BUCKET_ORDER:
        return -1.00

    target_bucket = target_bucket_for_action(workout_type, user_state)

    if target_bucket not in DISTANCE_BUCKET_ORDER:
        return 0.00

    target_position = DISTANCE_BUCKET_ORDER.index(target_bucket)
    bucket_position = DISTANCE_BUCKET_ORDER.index(distance_bucket)
    distance = abs(bucket_position - target_position)

    if distance == 0:
        return 1.00
    if distance == 1:
        return 0.35
    return -0.55 * distance


def readiness_type_score(workout_type, user_state):
    readiness_state = user_state["readiness_state"]

    if readiness_state == "return_to_activity":
        return 1.20 if workout_type == "easy" else -2.00

    if readiness_state == "base_building":
        return {
            "easy": 0.85,
            "tempo": 0.15,
            "intervals": -1.00,
            "long": -0.75,
        }.get(workout_type, 0)

    return {
        "easy": 0.25,
        "tempo": 0.25,
        "intervals": 0.25,
        "long": 0.25,
    }.get(workout_type, 0)


def recovery_type_score(workout_type, user_state, objective):
    fatigue_state = user_state["fatigue_state"]
    recovery_state = user_state["recovery_state"]
    readiness_state = user_state["readiness_state"]

    score = 0

    if fatigue_state == "high_fatigue":
        score += 0.90 if workout_type == "easy" else -1.20

    if recovery_state == "recent_intensity":
        score += 0.75 if workout_type == "easy" else -1.00

    if recovery_state == "moderate_recovery" and workout_type == "intervals":
        score -= 0.20

    if recovery_state == "recovered" and readiness_state == "active_runner":
        if objective == "Melhorar velocidade" and workout_type in ["tempo", "intervals"]:
            score += 0.45
        elif objective == "Construir endurance" and workout_type == "long":
            score += 0.45
        elif objective == "Ganhar consistência" and workout_type == "easy":
            score += 0.25

    return score


def variety_score(workout_type, user_state):
    last_workout_type = user_state["last_workout_type"]

    if last_workout_type == "rest":
        return 0.15

    if workout_type == last_workout_type:
        return -0.25

    if last_workout_type in ["tempo", "intervals", "long"] and workout_type == "easy":
        return 0.35

    return 0.10


def plan_prior_score(action, plan_prior):
    return 0.25 * float(plan_prior.get(action, 0))


def global_prior_score(action, global_ranking):
    if action not in global_ranking:
        return 0

    position = global_ranking.index(action)
    return 0.05 * (1 / (position + 1))


def is_action_allowed_by_readiness(action, user_state):
    workout_type = action_workout_type(action)
    distance_bucket = action_distance_bucket(action)
    readiness_state = user_state["readiness_state"]

    if action == REST_ACTION:
        return readiness_state == "return_to_activity"

    if readiness_state == "return_to_activity":
        return workout_type == "easy" and distance_bucket == "very_short"

    if readiness_state == "base_building":
        if workout_type == "easy" and distance_bucket in ["very_short", "short", "medium"]:
            return True
        if workout_type == "tempo" and distance_bucket == "short":
            return user_state["recovery_state"] == "recovered"
        return False

    return True


def is_action_allowed_by_safety(action, user_state):
    if action == REST_ACTION:
        return True

    workout_type = action_workout_type(action)

    high_fatigue = user_state["prev_tsb"] <= TSB_THRESHOLD
    recent_intensity = user_state["days_since_intense"] < 2
    after_inactive_week = user_state["prev_is_inactive_week"]

    if high_fatigue and workout_type != "easy":
        return False

    if recent_intensity and workout_type in ["tempo", "intervals"]:
        return False

    if recent_intensity and action == "long__very_long":
        return False

    if after_inactive_week and workout_type in ["intervals", "long"]:
        return False

    return True


def should_offer_rest(user_state):
    if user_state["readiness_state"] == "return_to_activity" and user_state["prev_weekly_km"] == 0:
        return True

    high_fatigue = user_state["prev_tsb"] <= TSB_THRESHOLD
    recent_intensity = user_state["days_since_intense"] < 2
    inactive_week = user_state["prev_is_inactive_week"]

    return high_fatigue and (recent_intensity or inactive_week)


def get_allowed_actions(user_state, action_space):
    allowed = [
        action for action in action_space
        if is_action_allowed_by_readiness(action, user_state)
        and is_action_allowed_by_safety(action, user_state)
    ]

    if should_offer_rest(user_state) and REST_ACTION not in allowed:
        allowed.append(REST_ACTION)

    if not allowed:
        return ["easy__very_short", REST_ACTION]

    return allowed


def rank_actions(action_space, user_state, objective, plan_prior, global_ranking):
    rows = []
    objective_profile = OBJECTIVE_PROFILES[objective]

    for action in action_space:
        workout_type = action_workout_type(action)
        distance_bucket = action_distance_bucket(action)

        goal_component = 2.20 * objective_profile.get(workout_type, 0)
        readiness_component = readiness_type_score(workout_type, user_state)
        recovery_component = recovery_type_score(workout_type, user_state, objective)
        distance_component = 1.10 * distance_fit_score(
            workout_type,
            distance_bucket,
            user_state,
        )
        variety_component = variety_score(workout_type, user_state)
        plan_component = plan_prior_score(action, plan_prior)
        global_component = global_prior_score(action, global_ranking)

        score = (
            goal_component
            + readiness_component
            + recovery_component
            + distance_component
            + variety_component
            + plan_component
            + global_component
        )

        rows.append({
            "action_id": action,
            "score": score,
            "goal_score": goal_component,
            "readiness_score": readiness_component,
            "recovery_score": recovery_component,
            "distance_score": distance_component,
            "variety_score": variety_component,
            "plan_prior_score": plan_component,
            "global_prior_score": global_component,
            "target_bucket": target_bucket_for_action(workout_type, user_state),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["score", "goal_score", "distance_score", "action_id"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )


def build_final_ranking(base_ranking_df, allowed_actions):
    allowed_set = set(allowed_actions)
    final_actions = [
        action for action in base_ranking_df["action_id"].tolist()
        if action in allowed_set
    ]

    if REST_ACTION in allowed_actions and REST_ACTION not in final_actions:
        final_actions.append(REST_ACTION)

    return final_actions[:TOP_K]


def build_reason_code(user_state, ranking_changed):
    reasons = []

    if user_state["readiness_state"] == "return_to_activity":
        reasons.append("retorno gradual à atividade")

    if user_state["readiness_state"] == "base_building":
        reasons.append("reconstrução de base")

    if user_state["prev_tsb"] <= TSB_THRESHOLD:
        reasons.append("fadiga elevada")

    if user_state["days_since_intense"] < 2:
        reasons.append("intensidade recente")

    if user_state["prev_is_inactive_week"]:
        reasons.append("semana anterior inativa")

    if not reasons:
        return "Ranking preservado: não há gatilhos contextuais de risco."

    prefix = "Ranking ajustado" if ranking_changed else "Ranking validado"
    return f"{prefix}: " + ", ".join(reasons) + "."


def build_goal_alignment_text(user_state, objective, primary_action):
    workout_type = action_workout_type(primary_action)

    if primary_action == REST_ACTION:
        return (
            "O descanso entrou como override operacional porque o estado atual "
            "indica risco alto para recomendar carga adicional."
        )

    if objective == "Melhorar velocidade":
        if user_state["readiness_state"] == "return_to_activity":
            return (
                "O objetivo é velocidade, mas a prontidão atual pede reconstrução "
                "de base antes de inserir intensidade."
            )

        if workout_type in ["tempo", "intervals"]:
            return (
                "A recomendação conversa diretamente com velocidade: ela prioriza "
                "um estímulo de intensidade compatível com recuperação e volume recente."
            )

        return (
            "Mesmo com objetivo de velocidade, o estado atual favorece um easy run. "
            "Isso protege recuperação e cria base para treinos intensos futuros."
        )

    if objective == "Construir endurance":
        if workout_type in ["long", "easy"]:
            return (
                "A recomendação conversa com endurance ao priorizar volume aeróbico "
                "e progressão de distância compatível com o histórico recente."
            )

        return (
            "O app preservou algum estímulo de qualidade, mas sem romper as travas "
            "de recuperação e progressão de volume."
        )

    return (
        "A recomendação conversa com consistência ao priorizar um treino sustentável, "
        "com menor chance de quebrar a sequência semanal."
    )


def format_action_list(actions):
    if not actions:
        return "Nenhuma"

    return "\n".join(
        f"{idx + 1}. `{action}` - {action_to_label(action)}"
        for idx, action in enumerate(actions)
    )


global_ranking = load_global_ranking()
plan_prior = load_plan_prior()
metrics = load_json_if_exists(METRICS_PATH)
safety_audit = load_json_if_exists(SAFETY_AUDIT_PATH)
safety_rules = load_json_if_exists(SAFETY_RULES_PATH)


st.title("RunRec MVP")
st.caption("Next Best Workout para um novo corredor")

st.info(
    "Esta versão separa decisão de treino em três camadas: objetivo do atleta, "
    "prontidão pelo volume recente e segurança fisiológica. O ranking global do "
    "experimento agora é apenas um desempate fraco, não o motor principal."
)


with st.sidebar:
    st.header("Seus dados")

    user_name = st.text_input("Nome ou apelido", value="Novo corredor")

    comfortable_pace = st.number_input(
        "Pace confortável atual (min/km)",
        min_value=3.0,
        max_value=12.0,
        value=6.5,
        step=0.1,
    )

    days_since_intense = st.number_input(
        "Dias desde o último treino intenso",
        min_value=0,
        max_value=30,
        value=4,
        step=1,
    )

    current_status = st.selectbox(
        "Status atual",
        [
            "Corro continuamente",
            "Retomando com caminhada + trote",
            "Não estou correndo / só caminhada",
        ],
    )

    weeks_without_running = st.number_input(
        "Semanas sem correr de forma consistente",
        min_value=0,
        max_value=104,
        value=0,
        step=1,
    )

    objective = st.selectbox(
        "Objetivo do próximo bloco",
        [
            "Melhorar velocidade",
            "Ganhar consistência",
            "Construir endurance",
        ],
    )

    last_workout_label = st.selectbox(
        "Último treino realizado",
        list(LAST_WORKOUT_LABELS.keys()),
    )

    st.caption(
        "Treino intenso = intervalado, tempo run, longão forte ou sessão que deixou fadiga clara."
    )


default_weeks = pd.DataFrame({
    "week": [
        "Semana -6",
        "Semana -5",
        "Semana -4",
        "Semana -3",
        "Semana -2",
        "Semana -1",
    ],
    "weekly_km": [18.0, 20.0, 22.0, 24.0, 26.0, 28.0],
    "intensity_profile": [
        "Predominantemente leve",
        "Misto / moderado",
        "Predominantemente leve",
        "Misto / moderado",
        "Predominantemente leve",
        "Predominantemente leve",
    ],
})

st.subheader("Histórico recente")
st.write(
    "Informe o volume das últimas 6 semanas. O app usa esse histórico para estimar ATL, CTL, TSB e prontidão."
)

weeks_df = st.data_editor(
    default_weeks,
    hide_index=True,
    use_container_width=True,
    column_config={
        "week": st.column_config.TextColumn("Semana", disabled=True),
        "weekly_km": st.column_config.NumberColumn(
            "Km na semana",
            min_value=0.0,
            max_value=200.0,
            step=1.0,
        ),
        "intensity_profile": st.column_config.SelectboxColumn(
            "Intensidade dominante",
            options=list(INTENSITY_FACTORS.keys()),
        ),
    },
)

user_state = build_user_state(
    weeks_df=weeks_df,
    comfortable_pace=comfortable_pace,
    days_since_intense=days_since_intense,
    current_status=current_status,
    weeks_without_running=weeks_without_running,
    last_workout_type=LAST_WORKOUT_LABELS[last_workout_label],
)

base_ranking_df = rank_actions(
    action_space=ACTION_SPACE,
    user_state=user_state,
    objective=objective,
    plan_prior=plan_prior,
    global_ranking=global_ranking,
)

allowed_actions = get_allowed_actions(user_state, ACTION_SPACE)
safe_ranking = build_final_ranking(base_ranking_df, allowed_actions)

base_top_k = base_ranking_df["action_id"].head(TOP_K).tolist()
ranking_changed = safe_ranking != base_top_k

removed_actions = [
    action for action in base_top_k
    if action not in safe_ranking
]

added_actions = [
    action for action in safe_ranking
    if action not in base_top_k
]

primary_action = safe_ranking[0]
primary_output = action_to_output(primary_action)


metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("TSB estimado", f"{user_state['prev_tsb']:.1f}")
metric_col2.metric("Estado de fadiga", user_state["fatigue_state"])
metric_col3.metric("Km médio / semana", f"{user_state['avg_weekly_km']:.1f}")
metric_col4.metric("Ranking alterado?", "Sim" if ranking_changed else "Não")

st.caption(f"Estado de prontidão: `{user_state['readiness_state']}`")

st.divider()

rec_col, reason_col = st.columns([1, 2])

with rec_col:
    st.subheader("Próximo treino")
    st.success(primary_output["tipo_treino"])

    st.metric(
        "Distância recomendada",
        primary_output["distancia_recomendada"],
    )

    st.caption(f"Action ID: `{primary_output['action_id']}`")

    st.markdown("#### Alternativas")

    if len(safe_ranking) == 1:
        st.caption(
            "Sem alternativas seguras neste cenário. "
            "Como o estado atual é de retorno, o app mantém apenas a opção mais conservadora."
        )
    else:
        alternatives_df = pd.DataFrame([
            {
                "prioridade": idx + 2,
                **action_to_output(action),
            }
            for idx, action in enumerate(safe_ranking[1:])
        ])

        st.dataframe(
            alternatives_df[
                [
                    "prioridade",
                    "tipo_treino",
                    "distancia_recomendada",
                    "action_id",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

with reason_col:
    st.subheader("Justificativa")
    st.write(
        f"{user_name}, a recomendação combina objetivo declarado, volume recente, "
        "recuperação, fadiga estimada e uma regra simples de variedade em relação ao último treino."
    )

    st.info(build_reason_code(user_state, ranking_changed))

    st.markdown("#### Alinhamento com objetivo")
    st.write(build_goal_alignment_text(user_state, objective, primary_action))

    st.markdown("#### Como o motor decidiu")
    st.write(
        "O objetivo define a preferência de estímulo; o volume recente ajusta a faixa de distância; "
        "a segurança remove treinos incompatíveis com fadiga, intensidade recente ou retorno gradual."
    )

st.divider()

rank_col1, rank_col2, rank_col3 = st.columns(3)

with rank_col1:
    st.subheader("Ranking antes da segurança")
    st.markdown(format_action_list(base_top_k))

with rank_col2:
    st.subheader("Ranking final")
    st.markdown(format_action_list(safe_ranking))

with rank_col3:
    st.subheader("Intervenções")
    st.markdown("**Removidas**")
    st.markdown(format_action_list(removed_actions))

    st.markdown("**Adicionadas**")
    st.markdown(format_action_list(added_actions))

st.divider()

st.subheader("Output estruturado da recomendação")

output_table = pd.DataFrame([
    {
        "prioridade": idx + 1,
        **action_to_output(action),
    }
    for idx, action in enumerate(safe_ranking)
])

st.dataframe(
    output_table[
        [
            "prioridade",
            "tipo_treino",
            "distancia_recomendada",
            "faixa_distancia",
            "action_id",
        ]
    ],
    hide_index=True,
    use_container_width=True,
)

if primary_action == REST_ACTION:
    st.warning(
        "Descanso aparece como override operacional de segurança. No notebook, descanso não é uma classe supervisionada "
        "aprendida na base, porque a base Kaggle contém apenas treinos realizados."
    )

st.divider()

state_col, chart_col = st.columns([1, 2])

with state_col:
    st.subheader("Contexto usado")

    context_table = pd.DataFrame({
        "campo": [
            "Objetivo",
            "Pace confortável",
            "Dias desde intensidade",
            "Último treino",
            "Semana anterior inativa?",
            "Status atual",
            "Semanas sem correr",
            "Prontidão",
            "Recuperação",
            "ATL estimado",
            "CTL estimado",
        ],
        "valor": [
            objective,
            f"{comfortable_pace:.1f} min/km",
            f"{user_state['days_since_intense']:.0f}",
            last_workout_label,
            "Sim" if user_state["prev_is_inactive_week"] else "Não",
            user_state["current_status"],
            str(user_state["weeks_without_running"]),
            user_state["readiness_state"],
            user_state["recovery_state"],
            f"{user_state['prev_atl']:.1f}",
            f"{user_state['prev_ctl']:.1f}",
        ],
    })

    st.dataframe(context_table, hide_index=True, use_container_width=True)

with chart_col:
    st.subheader("Carga recente estimada")

    chart_df = (
        user_state["state_df"][
            ["week", "weekly_trimp", "atl", "ctl", "tsb"]
        ]
        .set_index("week")
    )

    st.line_chart(chart_df)

with st.expander("Detalhe do score das ações"):
    display_df = base_ranking_df.copy()
    display_df["label"] = display_df["action_id"].apply(action_to_label)
    display_df["allowed_by_safety"] = display_df["action_id"].isin(allowed_actions)
    display_df["tipo_treino"] = display_df["action_id"].apply(
        lambda action: action_to_output(action)["tipo_treino"]
    )
    display_df["distancia_recomendada"] = display_df["action_id"].apply(
        lambda action: action_to_output(action)["distancia_recomendada"]
    )

    display_cols = [
        "action_id",
        "label",
        "score",
        "goal_score",
        "readiness_score",
        "recovery_score",
        "distance_score",
        "variety_score",
        "plan_prior_score",
        "global_prior_score",
        "target_bucket",
        "allowed_by_safety",
    ]

    st.dataframe(display_df[display_cols], use_container_width=True)

with st.expander("Métricas e auditoria do experimento"):
    if metrics is not None:
        st.markdown("#### Métricas")
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)
    else:
        st.write("Artefato `model_metrics.json` ainda não encontrado.")

    if safety_audit is not None:
        st.markdown("#### Auditoria da camada de segurança")
        st.dataframe(pd.DataFrame(safety_audit), use_container_width=True)
    else:
        st.write("Artefato `safety_audit.json` ainda não encontrado.")

    if safety_rules is not None:
        st.markdown("#### Regras exportadas")
        st.json(safety_rules)

st.caption(
    "Uso educacional e demonstrativo. O RunRec não substitui avaliação profissional, "
    "especialmente em caso de dor, lesão, pós-parto, gestação ou condição clínica."
)
