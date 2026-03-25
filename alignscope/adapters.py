from typing import Dict, Any

def extract_mpe_state(env, agent_id: str) -> Dict[str, Any]:
    """
    Extracts continuous (x, y) coordinates from PettingZoo MPE physics engine.
    MPE agents have state.p_pos = [x, y].
    """
    try:
        # Aces environments often expose the raw wrapped env
        raw_env = getattr(env, "env", env)
        if hasattr(raw_env, "unwrapped"):
            raw_env = raw_env.unwrapped
            
        if hasattr(raw_env, "world"):
            # Multi-Particle Environment (MPE)
            for internal_agent in raw_env.world.agents:
                # MPE raw agents are usually just named "agent 0", "adversary 0"
                if agent_id.replace("_", " ") in internal_agent.name or internal_agent.name.replace(" ", "_") in agent_id:
                    return {
                        "x": float(internal_agent.state.p_pos[0]),
                        "y": float(internal_agent.state.p_pos[1])
                    }
    except Exception:
        pass
    
    # Fallback if properties not found
    return {"x": 0.0, "y": 0.0}


def extract_kaz_state(env, agent_id: str) -> Dict[str, Any]:
    """
    Extracts grid coordinates from PettingZoo Knights-Archers-Zombies (KAZ).
    Also accurately identifies the role based on the agent name.
    """
    role = "agent"
    if "knight" in agent_id.lower():
        role = "knight"
    elif "archer" in agent_id.lower():
        role = "archer"
    elif "zombie" in agent_id.lower():
        role = "zombie"

    pos = {"x": 0.0, "y": 0.0}
    try:
        raw_env = getattr(env, "env", env)
        if hasattr(raw_env, "unwrapped"):
            raw_env = raw_env.unwrapped

        # KAZ internal objects are kept in lists like env.knights, env.archers
        collection = []
        if role == "knight" and hasattr(raw_env, "knights"):
            collection = raw_env.knights
        elif role == "archer" and hasattr(raw_env, "archers"):
            collection = raw_env.archers
        elif role == "zombie" and hasattr(raw_env, "zombies"):
            collection = raw_env.zombies

        # We try to match the index if agent_id ends with a number (e.g., "knight_0")
        try:
            idx = int(agent_id.split("_")[-1])
            if idx < len(collection):
                obj = collection[idx]
                if hasattr(obj, "position"): # Usually [x, y] in KAZ
                    pos["x"] = float(obj.position[0])
                    pos["y"] = float(obj.position[1])
        except (ValueError, IndexError):
            pass

    except Exception:
        pass

    return {
        "role": role,
        "x": pos["x"],
        "y": pos["y"]
    }

def extract_smac_state(env, agent_id: str) -> Dict[str, Any]:
    """
    Extracts unit type, position, health, and shield from SMAC (StarCraft Multi-Agent Challenge).
    
    SMAC exposes per-unit data via env.get_unit_by_id() or the internal
    controller's allies/enemies lists.
    """
    role = "marine"  # Default SMAC unit
    pos = {"x": 0.0, "y": 0.0}
    health = 0.0
    shield = 0.0

    try:
        raw_env = getattr(env, "env", env)
        if hasattr(raw_env, "unwrapped"):
            raw_env = raw_env.unwrapped

        # Extract agent index
        try:
            idx = int(agent_id.split("_")[-1])
        except (ValueError, IndexError):
            idx = 0

        # SMAC StarCraft2Env exposes allies via controller
        if hasattr(raw_env, "agents") and hasattr(raw_env, "get_unit_by_id"):
            unit = raw_env.get_unit_by_id(idx)
            if unit is not None:
                pos["x"] = float(unit.pos.x) if hasattr(unit, "pos") else 0.0
                pos["y"] = float(unit.pos.y) if hasattr(unit, "pos") else 0.0
                health = float(unit.health) / float(unit.health_max) if hasattr(unit, "health_max") and unit.health_max > 0 else 0.0
                shield = float(unit.shield) / float(unit.shield_max) if hasattr(unit, "shield_max") and unit.shield_max > 0 else 0.0

                # Infer role from unit type
                unit_type = getattr(unit, "unit_type", None)
                type_map = {
                    0: "marine", 1: "marauder", 2: "medivac",
                    48: "marine", 49: "marauder", 50: "reaper",
                    51: "ghost", 73: "zealot", 74: "stalker",
                    75: "sentry", 76: "high_templar", 77: "colossus",
                }
                if unit_type is not None:
                    role = type_map.get(unit_type, f"unit_{unit_type}")

        # Fallback: try the _obs property or controller
        elif hasattr(raw_env, "controller") or hasattr(raw_env, "_sc2_env"):
            # Newer SMAC versions
            sc2_env = getattr(raw_env, "_sc2_env", raw_env)
            if hasattr(sc2_env, "get_ally_units"):
                allies = sc2_env.get_ally_units()
                if idx < len(allies):
                    unit = allies[idx]
                    pos["x"] = float(getattr(unit, "x", 0))
                    pos["y"] = float(getattr(unit, "y", 0))
                    health = float(getattr(unit, "health", 0)) / max(float(getattr(unit, "health_max", 1)), 1)
                    role = "ally"

    except Exception:
        pass

    return {
        "role": role,
        "x": pos["x"],
        "y": pos["y"],
        "health": health,
        "shield": shield,
    }


def try_extract_env_state(env, agent_id: str) -> Dict[str, Any]:
    """
    Master adapter router. Detects environment type and routes to the correct extractor.
    """
    env_name = str(env).lower()
    
    # Check for MPE
    if "mpe" in env_name or "simple_" in env_name:
        return extract_mpe_state(env, agent_id)
        
    # Check for Knights Archers Zombies
    if "knights_archers_zombies" in env_name or "kaz" in env_name:
        return extract_kaz_state(env, agent_id)
    
    # Check for SMAC (StarCraft Multi-Agent Challenge)
    if "starcraft" in env_name or "smac" in env_name or "sc2" in env_name:
        return extract_smac_state(env, agent_id)

    # Check for SMAC by attribute detection (env_name may not contain keywords)
    raw = getattr(env, "unwrapped", env)
    if hasattr(raw, "get_unit_by_id") or hasattr(raw, "_sc2_env"):
        return extract_smac_state(env, agent_id)
        
    return {"x": 0.0, "y": 0.0}