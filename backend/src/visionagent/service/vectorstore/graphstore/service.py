import asyncio
import logging
import re
import time
from typing import Any

from visionagent.config.settings import settings
from visionagent.database.graph import GraphRepository, edge_vector_id
from visionagent.database.graph.provenance import (
    DOCUMENT_NAMES_FIELD,
    LegacyGraphProvenanceError,
    contribution_document_names,
    dump_document_names,
    load_document_names,
    source_ids,
)
from visionagent.models import ParsedChunk
from visionagent.providers.llm import build_llm

from .lightrag_utils import _merge_edges_then_upsert, _merge_nodes_then_upsert, compute_mdhash_id

logger = logging.getLogger(__name__)


def _node_vector_metadata(name: str, node: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "entity_name": name,
        "entity_type": node.get("entity_type", "unknown"),
        "content": f"{name}\n{node.get('description', '')}",
        "source_id": node.get("source_id", ""),
        "docnm": node.get("docnm", "unknown"),
    }
    document_names = contribution_document_names(node)
    if document_names is not None:
        metadata[DOCUMENT_NAMES_FIELD] = dump_document_names(document_names)
    return metadata


def _edge_vector_metadata(edge: dict[str, Any]) -> dict[str, Any]:
    source = str(edge["src_id"])
    target = str(edge["tgt_id"])
    keywords = str(edge.get("keywords", ""))
    metadata = {
        "src_id": source,
        "tgt_id": target,
        "keywords": keywords,
        "content": f"{source} {target} {keywords} {edge.get('description', '')}",
        "source_id": edge.get("source_id", ""),
        "docnm": edge.get("docnm", "unknown"),
        "weight": edge.get("weight", 1.0),
    }
    document_names = contribution_document_names(edge)
    if document_names is not None:
        metadata[DOCUMENT_NAMES_FIELD] = dump_document_names(document_names)
    return metadata

# Entity-extraction prompt adapted from LightRAG.
PROMPTS = {
    "entity_extraction": """---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use English as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name
- entity_type: One of the following types: [organization, person, geo, event, category]
- entity_description: Provide a comprehensive description of the entity's attributes and activities *based solely on the information present in the input text*. **Do not infer or hallucinate information not explicitly stated.** If the text provides insufficient information to create a comprehensive description, state "Description not available in text."
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"<|><high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

5. When finished, output <|COMPLETE|>

######################
---Examples---
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

Output:
("entity"<|>"Alex"<|>"person"<|>"Alex is a character who experiences frustration and is observant of the dynamics among other characters.")##
("entity"<|>"Taylor"<|>"person"<|>"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.")##
("entity"<|>"Jordan"<|>"person"<|>"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.")##
("entity"<|>"Cruz"<|>"person"<|>"Cruz is associated with a vision of control and order, influencing the dynamics among other characters.")##
("entity"<|>"The Device"<|>"technology"<|>"The Device is central to the story, with potential game-changing implications, and is revered by Taylor.")##
("relationship"<|>"Alex"<|>"Taylor"<|>"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."<|>"power dynamics, perspective shift"<|>7)##
("relationship"<|>"Alex"<|>"Jordan"<|>"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."<|>"shared goals, rebellion"<|>6)##
("relationship"<|>"Taylor"<|>"Jordan"<|>"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."<|>"conflict resolution, mutual respect"<|>8)##
("relationship"<|>"Jordan"<|>"Cruz"<|>"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."<|>"ideological conflict, rebellion"<|>5)##
("relationship"<|>"Taylor"<|>"The Device"<|>"Taylor shows reverence towards the device, indicating its importance and potential impact."<|>"reverence, technological significance"<|>9)##
("content_keywords"<|>"power dynamics, ideological conflict, discovery, rebellion")<|COMPLETE|>

#############################
---Real Data---
######################
Entity_types: [organization, person, geo, event, category]
Text:
{input_text}
######################
Output:"""
}

def _llm() -> Any:
    """Build a client for one service/event-loop lifetime.

    Upload workers call ``asyncio.run`` once per file. A module singleton would
    therefore retain an async HTTP transport bound to a closed event loop and
    reuse it in the next file's loop.
    """
    return build_llm()


class GraphRAGService:
    """Extract, merge, persist, and delete tenant-scoped graph facts."""
    
    def __init__(
        self,
        user_id: str,
        repository: GraphRepository | None = None,
        llm: Any | None = None,
    ) -> None:
        self.user_id = user_id

        # Storage lifetime belongs to the injected repository. The service
        # closes only resources it constructs itself.
        self._owns_repository = repository is None
        self._owns_llm = llm is None
        self._llm_client = llm
        self.repository = repository or GraphRepository(user_id)
        working_dir = self.repository.working_dir
        self.node_vdb = self.repository.node_vdb
        self.edge_vdb = self.repository.edge_vdb
        self.knowledge_graph = self.repository.knowledge_graph
        
        self.global_config = {"working_dir": working_dir}
        
    async def _call_llm(self, prompt: str) -> str:
        """Call the extraction model and propagate provider failures."""
        if self._llm_client is None:
            self._llm_client = _llm()
        return str(
            await self._llm_client.complete(prompt=prompt, model=settings.ner_model)
        )

    def _parse_lightrag_response(self, response: str, chunk_id: str, doc_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse LightRAG's structured response format."""
        entities = []
        relationships = []
        
        
        records = response.split("##")
        
        for _i, record in enumerate(records):
            record = record.strip()
            
            if not record or "<|COMPLETE|>" in record:
                continue
                
            match = re.search(r'\(([^)]+)\)', record)
            if not match:
                continue
                
            content = match.group(1)
            
            quoted_strings = re.findall(r'"([^"]*)"', content)
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', content)
            
            if len(quoted_strings) >= 3:
                record_type = quoted_strings[0]
                
                if record_type == "entity" and len(quoted_strings) >= 4:
                    entity_name = quoted_strings[1]
                    
                    if not entity_name.strip():
                        continue
                    
                    entity = {
                        "entity_name": entity_name,
                        "entity_type": quoted_strings[2],
                        "description": quoted_strings[3],
                        "source_id": chunk_id,
                        "docnm": doc_id,  
                    }
                    entities.append(entity)
                    
                elif record_type == "relationship" and len(quoted_strings) >= 5:
                    source = quoted_strings[1]
                    target = quoted_strings[2]
                    
                    
                    if not source.strip() or not target.strip():
                        continue
                    
                    weight = 1.0
                    if numbers:
                        try:
                            weight = float(numbers[0])
                        except ValueError:
                            weight = 1.0
                        
                    relationship = {
                        "source": source,
                        "target": target,
                        "src_id": source,  # LightRAG utils expect this field
                        "tgt_id": target,  # LightRAG utils expect this field
                        "description": quoted_strings[3],
                        "keywords": quoted_strings[4] if len(quoted_strings) > 4 else "",
                        "weight": weight,
                        "source_id": chunk_id,
                        "docnm": doc_id,  
                    }
                    relationships.append(relationship)
        
        return entities, relationships

    async def _process_entity_name(self, entity_name: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge and persist all contributions for one entity."""
        entity_data = await _merge_nodes_then_upsert(
            entity_name, entities, self.knowledge_graph,
            self.global_config, None, None, None
        )

        if self.node_vdb is not None:
            data_for_vdb = {
                compute_mdhash_id(entity_data["entity_name"], prefix="ent-"):
                    _node_vector_metadata(entity_data["entity_name"], entity_data)
            }
            await self.node_vdb.upsert(data_for_vdb)

        return entity_data

    async def _process_edges(self, edge_key: tuple[str, str], edges: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[Any]]:
        """Merge and persist all contributions for one relationship."""
        edge_data = await _merge_edges_then_upsert(
            edge_key[0], edge_key[1], edges, self.knowledge_graph,
            self.global_config, None, None, None, []
        )

        if self.edge_vdb is not None and edge_data:
            data_for_vdb = {
                edge_vector_id(
                    edge_data["src_id"], edge_data["tgt_id"]
                ): _edge_vector_metadata(edge_data)
            }
            await self.edge_vdb.upsert(data_for_vdb)

        return edge_data, []

    async def aclose(self) -> None:
        """Attempt every owned close, even when one transport refuses."""
        closes = []
        if self._owns_llm and self._llm_client is not None:
            close = getattr(self._llm_client, "aclose", None)
            if close is not None:
                closes.append(close())
            self._llm_client = None
        if self._owns_repository:
            closes.append(self.repository.aclose())
        if not closes:
            return

        results = await asyncio.gather(*closes, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            # All resources were attempted. Surface the first cleanup failure;
            # the public wrappers below preserve an already-active processing
            # exception instead of replacing it with this one.
            raise failures[0]

    async def process_chunks_for_graphrag(
        self,
        chunks: list[ParsedChunk],
        file_name: str,
        callback: Any = None,
    ) -> dict[str, Any]:
        """Process one document and release any owned async transports."""
        primary_error: BaseException | None = None
        try:
            return await self._process_chunks_for_graphrag(
                chunks,
                file_name,
                callback,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await self.aclose()
            except BaseException:
                if primary_error is None:
                    raise
                logger.exception(
                    "GraphRAG resource cleanup failed after processing failure"
                )

    async def _process_chunks_for_graphrag(
        self,
        chunks: list[ParsedChunk],
        file_name: str,
        callback: Any = None,
    ) -> dict[str, Any]:
        """Extract and persist graph facts from normalized chunks."""
        
        if not chunks:
            return {"success": False, "entities": 0, "relationships": 0}
        
        start_time = time.time()
        
        all_entities = []
        all_relationships = []
        skipped_chunks = 0
        
        for _i, chunk in enumerate(chunks):
            # One LLM call per chunk, so this loop is the bulk of the run.
            # Reported at the top -- several paths below `continue` -- on the
            # caller's 0.3..0.5 band, which is the graph half of the encode step.
            if callback:
                callback(0.3 + (_i / len(chunks)) * 0.2,
                         f"Extracting entities from chunk {_i + 1}/{len(chunks)}")

            text = chunk.content
            chunk_id = chunk.id
            # Parser chunks intentionally carry no persisted document name.
            # The upload's original name is authoritative and is also what
            # delete_file_data matches against graph metadata.
            docnm = file_name
            
            if not text or not chunk_id:
                logger.warning("GraphRAG skipped an invalid chunk")
                continue
            
            word_count = len(text.split())
            if word_count < 10:
                logger.debug(
                    "GraphRAG skipped short chunk chunk_id=%s words=%d minimum=%d",
                    chunk_id,
                    word_count,
                    10,
                )
                skipped_chunks += 1
                continue
            
            prompt = PROMPTS["entity_extraction"].format(input_text=text)
            response = await self._call_llm(prompt)

            entities, relationships = self._parse_lightrag_response(response, chunk_id, docnm)
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        entity_groups: dict[str, list[Any]] = {}
        for entity in all_entities:
            name = entity["entity_name"]
            if name not in entity_groups:
                entity_groups[name] = []
            entity_groups[name].append(entity)
        
        relationship_groups: dict[tuple[str, str], list[Any]] = {}
        for rel in all_relationships:
            key = (rel["source"], rel["target"])
            if key not in relationship_groups:
                relationship_groups[key] = []
            relationship_groups[key].append(rel)
        
        processed_entities = []
        for entity_name, entities in entity_groups.items():
            entity_data = await self._process_entity_name(entity_name, entities)
            processed_entities.append(entity_data)
        
        processed_relationships = []
        for edge_key, edges in relationship_groups.items():
            edge_data, _added_entities = await self._process_edges(edge_key, edges)
            if edge_data:
                processed_relationships.append(edge_data)
        
        # Atomic replacement prevents readers from observing partial graph files.
        self.repository.save()

        processing_time = time.time() - start_time

        logger.info(
            "GraphRAG processing complete entities=%d relationships=%d "
            "chunks_processed=%d chunks_total=%d chunks_skipped=%d duration_s=%.2f",
            len(processed_entities),
            len(processed_relationships),
            len(chunks) - skipped_chunks,
            len(chunks),
            skipped_chunks,
            processing_time,
        )
        
        return {
            "success": True,
            "entities": len(processed_entities),
            "relationships": len(processed_relationships),
            "raw_entities": len(all_entities),
            "raw_relationships": len(all_relationships),
            "skipped_chunks": skipped_chunks,
            "total_chunks": len(chunks),
            "processing_time": processing_time
        }

    async def delete_file_data(self, file_name: str) -> dict[str, Any]:
        """Delete one document and release any owned async transports."""
        primary_error: BaseException | None = None
        try:
            return await self._delete_file_data(file_name)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await self.aclose()
            except BaseException:
                if primary_error is None:
                    raise
                logger.exception(
                    "GraphRAG resource cleanup failed after deletion failure"
                )

    async def _delete_file_data(self, file_name: str) -> dict[str, Any]:
        """Remove one document's contributions while retaining shared facts."""
        
        # The deletion response retains the legacy stable document identifier.
        import xxhash
        doc_id = xxhash.xxh64((file_name + str(self.user_id)).encode("utf-8")).hexdigest()
        
        def document_identity(metadata: dict[str, Any]) -> tuple[set[str], bool]:
            """Return names and whether they came from lossless provenance."""
            exact = load_document_names(metadata)
            if exact is not None:
                return exact, True
            return source_ids(str(metadata.get("docnm", ""))), False

        def references_document(metadata: dict[str, Any]) -> bool:
            names, lossless = document_identity(metadata)
            if lossless:
                return file_name in names
            # Raw equality catches legacy single-document names that contain
            # the display delimiter or edge whitespace. If splitting makes
            # that identity ambiguous, the missing-graph preflight below
            # fails closed rather than guessing.
            return (
                str(metadata.get("docnm", "")) == file_name
                or file_name in names
            )

        # A legacy orphaned *shared* VDB aggregate has no graph ledger from
        # which it can be rebuilt. Fail before mutating anything rather than
        # leaking the target or destroying another document's vector.
        for metadata in self.node_vdb.metadatas:
            if not references_document(metadata):
                continue
            document_names, lossless = document_identity(metadata)
            entity_name = metadata.get("entity_name")
            if (
                isinstance(entity_name, str)
                and not await self.knowledge_graph.has_node(entity_name)
                and not lossless
                and len(document_names) > 1
            ):
                raise LegacyGraphProvenanceError(
                    f"orphaned legacy node vector for {file_name!r} requires reindexing"
                )
        for metadata in self.edge_vdb.metadatas:
            if not references_document(metadata):
                continue
            document_names, lossless = document_identity(metadata)
            source, target = metadata.get("src_id"), metadata.get("tgt_id")
            if (
                isinstance(source, str)
                and isinstance(target, str)
                and not await self.knowledge_graph.has_edge(source, target)
                and not lossless
                and len(document_names) > 1
            ):
                raise LegacyGraphProvenanceError(
                    f"orphaned legacy edge vector for {file_name!r} requires reindexing"
                )

        changes = await self.knowledge_graph.remove_document(file_name)
        deleted_nodes = list(changes["deleted_nodes"])
        deleted_edges = dict(changes["deleted_edges"])
        self.node_vdb.delete_ids(
            [compute_mdhash_id(name, prefix="ent-") for name in deleted_nodes]
        )
        self.edge_vdb.delete_ids(
            [
                edge_vector_id(
                    str(edge.get("src_id", source)),
                    str(edge.get("tgt_id", target)),
                )
                for (source, target), edge in deleted_edges.items()
            ]
        )
        for name, node in changes["updated_nodes"].items():
            await self.node_vdb.upsert({
                compute_mdhash_id(name, prefix="ent-"): _node_vector_metadata(name, node)
            })
        for (_source, _target), edge in changes["updated_edges"].items():
            await self.edge_vdb.upsert({
                edge_vector_id(
                    edge["src_id"], edge["tgt_id"]
                ): _edge_vector_metadata(edge)
            })
        orphan_node_ids = []
        for item_id, metadata in zip(
            tuple(self.node_vdb.ids), tuple(self.node_vdb.metadatas), strict=True
        ):
            entity_name = metadata.get("entity_name")
            if not references_document(metadata) or not isinstance(entity_name, str):
                continue
            node = await self.knowledge_graph.get_node(entity_name)
            if node is None:
                orphan_node_ids.append(item_id)
                continue
            canonical_id = compute_mdhash_id(entity_name, prefix="ent-")
            await self.node_vdb.upsert(
                {canonical_id: _node_vector_metadata(entity_name, node)}
            )
            if item_id != canonical_id:
                orphan_node_ids.append(item_id)
        orphan_edge_ids = []
        for item_id, metadata in zip(
            tuple(self.edge_vdb.ids), tuple(self.edge_vdb.metadatas), strict=True
        ):
            source, target = metadata.get("src_id"), metadata.get("tgt_id")
            if (
                not references_document(metadata)
                or not isinstance(source, str)
                or not isinstance(target, str)
            ):
                continue
            edge = await self.knowledge_graph.get_edge(source, target)
            if edge is None:
                orphan_edge_ids.append(item_id)
                continue
            edge.setdefault("src_id", source)
            edge.setdefault("tgt_id", target)
            canonical_id = edge_vector_id(edge["src_id"], edge["tgt_id"])
            await self.edge_vdb.upsert(
                {canonical_id: _edge_vector_metadata(edge)}
            )
            if item_id != canonical_id:
                orphan_edge_ids.append(item_id)
        self.node_vdb.delete_ids(orphan_node_ids)
        self.edge_vdb.delete_ids(orphan_edge_ids)
        self.repository.save()

        result = {
            "success": True,
            "error": None,
            "file_name": file_name,
            "doc_id": doc_id,
            "entities_deleted": len(deleted_nodes),
            "relationships_deleted": len(deleted_edges),
            "graph_nodes_deleted": len(deleted_nodes),
            "graph_edges_deleted": len(deleted_edges),
            "graph_nodes_updated": len(changes["updated_nodes"]),
            "graph_edges_updated": len(changes["updated_edges"]),
        }
        
        return result
