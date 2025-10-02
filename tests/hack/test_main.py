from hack.main import agent


def test_agent_creation():
    """Test that an agent can be created with basic configuration."""
    assert agent.name == "Assistant"
    assert agent.instructions == "You are a helpful assistant"
