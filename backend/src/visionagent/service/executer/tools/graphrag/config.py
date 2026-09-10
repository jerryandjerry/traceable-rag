"""Graph retrieval limits owned by the tool that applies them."""

from visionagent.config.settings import settings


class GraphRAGConfig:
    """Expose canonical settings through the graph tool's configuration seam."""

    @property
    def ENTITY_TOP_K(self) -> int:  # noqa: N802 -- existing call sites
        return settings.graph_entity_top_k

    @property
    def RELATIONSHIP_TOP_K(self) -> int:  # noqa: N802
        return settings.graph_relation_top_k

config = GraphRAGConfig()
