"""Forensic Graph Transformation & Hierarchical Topology Engine.

Transforms raw, dense transaction events into an enterprise-grade forensic threat graph
(inspired by Palantir Gotham, Quantexa, and Chainalysis Reactor):
1. Hierarchical Equivalence Rollup: Contracts thousands of degree-1 leaf cards into
   Breach Campaign Nodes (R ~ log10(N_cards)), eliminating the D3 star-graph dandelion collapse.
2. Forensic Bridge Card Elevation: Identifies and elevates multi-syndicate pivot cards into
   prominent Bridge Card Nodes (amber diamond nodes).
3. Weighted Edge Consolidation: Collapses dense parallel transaction links into single weighted,
   logarithmic-scaled directional conduits with quadratic Bézier arc routing.
4. Multi-Focal Orbital Constellations: Computes radial gravity anchors per syndicate group,
   preventing all nodes from collapsing into a single central blob.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Dict, List, Set, Tuple


class ForensicGraphTransformer:
    """Transforms raw transaction records into a stratified, forensic threat intelligence graph."""

    def __init__(
        self,
        canvas_width: int = 1400,
        canvas_height: int = 900,
        max_unrolled_cards_per_campaign: int = 4,
        max_bridge_cards: int = 18,
        max_merchants: int = 60,
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.max_unrolled_cards_per_campaign = max_unrolled_cards_per_campaign
        self.max_bridge_cards = max_bridge_cards
        self.max_merchants = max_merchants

    def transform(self, fraud_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes raw fraud records into a structured forensic node-link threat graph."""
        if not fraud_records:
            return {"nodes": [], "links": [], "summary": self._empty_summary()}

        # 1. First Pass: Analyze card degrees and syndicate/merchant associations
        card_syndicates: Dict[str, Set[str]] = defaultdict(set)
        card_botnets: Dict[str, Set[str]] = defaultdict(set)
        card_merchants: Dict[str, Set[str]] = defaultdict(set)
        card_mules: Dict[str, Set[str]] = defaultdict(set)
        card_tx_counts: Dict[str, int] = defaultdict(int)
        card_volumes: Dict[str, float] = defaultdict(float)
        card_approvals: Dict[str, int] = defaultdict(int)
        card_scenarios: Dict[str, Set[str]] = defaultdict(set)
        card_metadata: Dict[str, Dict[str, Any]] = {}
        merchant_tx_counts: Dict[str, int] = defaultdict(int)

        syndicate_meta: Dict[str, Dict[str, Any]] = {}
        botnet_meta: Dict[str, Dict[str, Any]] = {}
        merchant_meta: Dict[str, Dict[str, Any]] = {}
        mule_meta: Dict[str, Dict[str, Any]] = {}

        for r in fraud_records:
            cid = str(r.get("card_id", ""))
            sid = str(r.get("syndicate_id", ""))
            bid = str(r.get("botnet_cluster_id", ""))
            mid = str(r.get("merchant_id", ""))
            mule_id = str(r.get("beneficiary_account_id", ""))
            amt = float(r.get("amount", 0.0))
            resp = str(r.get("response_code", "00"))
            is_approved = (resp == "00" or resp == "10")

            if not cid:
                continue

            card_tx_counts[cid] += 1
            card_volumes[cid] += amt
            if is_approved:
                card_approvals[cid] += 1
            scen = str(r.get("scenario_tag", ""))
            if scen:
                card_scenarios[cid].add(scen)

            if sid:
                card_syndicates[cid].add(sid)
                if sid not in syndicate_meta:
                    syndicate_meta[sid] = {
                        "id": sid,
                        "name": str(r.get("syndicate_name", sid)),
                        "archetype": str(r.get("syndicate_archetype", "Threat Group")),
                    }
            if bid:
                card_botnets[cid].add(bid)
                if bid not in botnet_meta:
                    botnet_meta[bid] = {
                        "id": bid,
                        "name": bid,
                        "syndicate_id": sid,
                        "subnet": str(r.get("ip_subnet_prefix", "N/A")),
                        "asn": str(r.get("asn", "N/A")),
                        "isp": str(r.get("isp", "N/A")),
                        "device": str(r.get("device_fingerprint_id", "N/A")),
                    }
            if mid:
                card_merchants[cid].add(mid)
                merchant_tx_counts[mid] += 1
                if mid not in merchant_meta:
                    merchant_meta[mid] = {
                        "id": mid,
                        "mcc": r.get("mcc", "N/A"),
                        "category": r.get("merchant_category", "N/A"),
                        "channel": r.get("channel_type", "N/A"),
                    }
            if mule_id:
                card_mules[cid].add(mule_id)
                if mule_id not in mule_meta:
                    mule_meta[mule_id] = {
                        "id": mule_id,
                        "tier": str(r.get("mule_tier", "TIER_1_SMURF")),
                        "channel": str(r.get("liquidation_channel", "P2P")),
                        "syndicate_id": sid,
                    }

            if cid not in card_metadata:
                card_metadata[cid] = {
                    "currency": r.get("currency", "USD"),
                    "vaai_score": r.get("vaai_score", "N/A"),
                    "latest_response": f"ISO {resp}",
                }

        # 2. Partition Cards into True Pivot Cards vs. Leaf Campaign Cards
        # Cross-syndicate pivot cards are strictly ranked to prevent star-graph fan-out collapse
        candidate_bridges = []
        leaf_cards: Set[str] = set()

        for cid, syns in card_syndicates.items():
            if len(syns) > 1:
                candidate_bridges.append((len(syns), card_tx_counts[cid], card_volumes[cid], cid))
            else:
                leaf_cards.add(cid)

        candidate_bridges.sort(reverse=True)
        bridge_cards: Set[str] = set(c for _, _, _, c in candidate_bridges[:self.max_bridge_cards])
        for _, _, _, c in candidate_bridges[self.max_bridge_cards:]:
            leaf_cards.add(c)

        # 3. Form Breach Campaigns and Unroll Active Cardholders
        # Group leaf cards by primary (syndicate_id, botnet_id, primary_scenario)
        campaign_groups: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        for cid in leaf_cards:
            syn_id = next(iter(card_syndicates[cid])) if card_syndicates[cid] else "SYN_UNKNOWN"
            bot_id = next(iter(card_botnets[cid])) if card_botnets[cid] else "BOT_DIRECT"
            scen = next(iter(card_scenarios[cid])) if card_scenarios[cid] else "EXPLOIT"
            campaign_groups[(syn_id, bot_id, scen)].append(cid)

        # 4. Compute Constellation Coordinate Anchors
        unique_syndicates = sorted(list(syndicate_meta.keys()))
        if not unique_syndicates:
            unique_syndicates = ["SYN_DEFAULT"]
            syndicate_meta["SYN_DEFAULT"] = {"id": "SYN_DEFAULT", "name": "Primary Threat Group", "archetype": "Default"}

        K = len(unique_syndicates)
        cx = self.canvas_width / 2.0
        cy = self.canvas_height / 2.0

        syndicate_anchors: Dict[str, Tuple[float, float]] = {}
        if K == 1:
            syndicate_anchors[unique_syndicates[0]] = (round(cx - cx * 0.35, 1), round(cy, 1))
        elif K == 2:
            # Widescreen lateral distribution: Left and Right with safe margins
            syndicate_anchors[unique_syndicates[0]] = (round(cx - cx * 0.42, 1), round(cy, 1))
            syndicate_anchors[unique_syndicates[1]] = (round(cx + cx * 0.42, 1), round(cy, 1))
        else:
            rx = cx * 0.54
            ry = cy * 0.44
            for idx, sid in enumerate(unique_syndicates):
                angle = 2.0 * math.pi * idx / K
                ax = cx + rx * math.cos(angle)
                ay = cy + ry * math.sin(angle)
                syndicate_anchors[sid] = (round(ax, 1), round(ay, 1))

        nodes: List[Dict[str, Any]] = []
        node_ids: Set[str] = set()

        def register_node(node_dict: Dict[str, Any]) -> None:
            nid = node_dict["id"]
            if nid not in node_ids:
                node_ids.add(nid)
                nodes.append(node_dict)

        # Register Syndicate Center Nodes
        for sid, meta in syndicate_meta.items():
            ax, ay = syndicate_anchors.get(sid, (cx, cy))
            register_node({
                "id": sid,
                "label": meta.get("name", sid),
                "type": "syndicate",
                "radius": 22.0,
                "archetype": meta.get("archetype", "Threat Group"),
                "syndicate_id": sid,
                "target_x": ax,
                "target_y": ay,
                "details": {
                    "Syndicate ID": sid,
                    "Name": meta.get("name", sid),
                    "Archetype": meta.get("archetype", "N/A"),
                },
            })

        # Register Botnet / Proxy Nodes
        botnet_angle_step = 2.0 * math.pi / 8.0
        for b_idx, (bid, meta) in enumerate(botnet_meta.items()):
            sid = meta.get("syndicate_id") or (unique_syndicates[0] if unique_syndicates else "")
            sax, say = syndicate_anchors.get(sid, (cx, cy))
            b_angle = b_idx * botnet_angle_step
            bx = sax + 80.0 * math.cos(b_angle)
            by = say + 80.0 * math.sin(b_angle)
            register_node({
                "id": bid,
                "label": f"Proxy Pool: {meta.get('isp', bid)}",
                "type": "botnet",
                "radius": 14.0,
                "syndicate_id": sid,
                "target_x": round(bx, 1),
                "target_y": round(by, 1),
                "details": {
                    "Cluster ID": bid,
                    "Subnet Prefix": meta.get("subnet", "N/A"),
                    "ASN": meta.get("asn", "N/A"),
                    "ISP": meta.get("isp", "N/A"),
                    "Device Signature": meta.get("device", "N/A"),
                },
            })

        # Register Unrolled Cardholders & Breach Campaign Remainder Nodes
        card_to_campaign_map: Dict[str, str] = {}
        unrolled_card_ids: Set[str] = set()

        for c_idx, ((sid, bid, scen), cards) in enumerate(campaign_groups.items()):
            # Sort cards by activity (fraud tx count and volume)
            sorted_cards = sorted(
                cards,
                key=lambda c: (card_tx_counts[c], card_volumes[c]),
                reverse=True,
            )
            # Unroll top active compromised cardholders as first-class nodes
            num_unroll = min(len(sorted_cards), self.max_unrolled_cards_per_campaign)
            cards_to_unroll = sorted_cards[:num_unroll]
            remainder_cards = sorted_cards[num_unroll:]

            sax, say = syndicate_anchors.get(sid, (cx, cy))
            camp_angle = (c_idx * 0.9) + 0.3

            # Register unrolled cardholder nodes
            for u_idx, cid in enumerate(cards_to_unroll):
                unrolled_card_ids.add(cid)
                u_offset_angle = camp_angle + (u_idx - num_unroll / 2.0) * 0.25
                card_x = sax + 140.0 * math.cos(u_offset_angle)
                card_y = say + 140.0 * math.sin(u_offset_angle)

                c_vol = card_volumes[cid]
                c_tx = card_tx_counts[cid]
                c_appr = card_approvals[cid]
                c_rate = round((c_appr / max(1, c_tx)) * 100.0, 1)
                c_meta = card_metadata.get(cid, {})
                c_merchs = sorted(list(card_merchants[cid]))

                register_node({
                    "id": cid,
                    "label": f"Card: {cid[-6:] if len(cid) >= 6 else cid}",
                    "type": "card",
                    "radius": 9.5,
                    "syndicate_id": sid,
                    "botnet_id": bid,
                    "target_x": round(card_x, 1),
                    "target_y": round(card_y, 1),
                    "details": {
                        "Cardholder ID": cid,
                        "Type": "COMPROMISED CARDHOLDER",
                        "Breach Vector": scen,
                        "Controlling Botnet": bid,
                        "Affiliated Syndicate": sid,
                        "Tx Attempts": c_tx,
                        "Fraud Volume": f"${c_vol:,.2f}",
                        "Approval Rate": f"{c_rate}%",
                        "Latest Response": c_meta.get("latest_response", "N/A"),
                        "Targeted Merchants": ", ".join(c_merchs[:8]) if c_merchs else "N/A",
                    },
                })

            # If there are remainder cards, register a breach campaign batch node
            if remainder_cards:
                camp_id = f"CAMP_{bid}_{c_idx:02d}"
                for c in remainder_cards:
                    card_to_campaign_map[c] = camp_id

                n_rem = len(remainder_cards)
                tot_vol = sum(card_volumes[c] for c in remainder_cards)
                tot_tx = sum(card_tx_counts[c] for c in remainder_cards)
                tot_appr = sum(card_approvals[c] for c in remainder_cards)
                appr_rate = round((tot_appr / max(1, tot_tx)) * 100.0, 1)

                camp_x = sax + 175.0 * math.cos(camp_angle)
                camp_y = say + 175.0 * math.sin(camp_angle)
                r_scale = min(24.0, max(11.0, 8.0 + 4.5 * math.log10(max(1, n_rem))))

                constituent_cards_data = [
                    {
                        "id": c,
                        "label": f"Card: {c[-6:] if len(c) >= 6 else c}",
                        "volume": round(card_volumes[c], 2),
                        "tx_count": card_tx_counts[c],
                        "approval_rate": round((card_approvals[c] / max(1, card_tx_counts[c])) * 100.0, 1),
                        "merchants": sorted(list(card_merchants[c]))[:5],
                    }
                    for c in remainder_cards
                ]

                register_node({
                    "id": camp_id,
                    "label": f"Batch: {n_rem} Cards",
                    "type": "breach_campaign",
                    "radius": round(r_scale, 1),
                    "card_count": n_rem,
                    "total_volume_usd": round(tot_vol, 2),
                    "approval_rate": appr_rate,
                    "syndicate_id": sid,
                    "botnet_id": bid,
                    "target_x": round(camp_x, 1),
                    "target_y": round(camp_y, 1),
                    "constituent_cards": constituent_cards_data,
                    "details": {
                        "Campaign ID": camp_id,
                        "Compromised Cards": f"{n_rem:,}",
                        "Total Extracted Volume": f"${tot_vol:,.2f}",
                        "Tx Attempts": tot_tx,
                        "Approval Rate": f"{appr_rate}%",
                        "Primary Vector": scen,
                        "Parent Botnet": bid,
                    },
                })

        # Register Elevated Bridge Cards (First-Class Glowing Amber Diamonds)
        for b_idx, cid in enumerate(sorted(list(bridge_cards))):
            syns = sorted(list(card_syndicates[cid]))
            merchs = sorted(list(card_merchants[cid]))
            # Place bridge card in the gravitational interstitial zone between its syndicates
            if len(syns) >= 2 and syns[0] in syndicate_anchors and syns[1] in syndicate_anchors:
                p1 = syndicate_anchors[syns[0]]
                p2 = syndicate_anchors[syns[1]]
                bx = (p1[0] + p2[0]) / 2.0
                by = (p1[1] + p2[1]) / 2.0
            elif syns and syns[0] in syndicate_anchors:
                sax, say = syndicate_anchors[syns[0]]
                bx = (sax + cx) / 2.0 + (b_idx * 15.0 % 50.0 - 25.0)
                by = (say + cy) / 2.0 + (b_idx * 15.0 % 50.0 - 25.0)
            else:
                bx = cx
                by = cy

            vol = card_volumes[cid]
            tx_cnt = card_tx_counts[cid]
            appr = card_approvals[cid]
            appr_pct = round((appr / max(1, tx_cnt)) * 100.0, 1)
            c_meta = card_metadata.get(cid, {})

            register_node({
                "id": cid,
                "label": f"Pivot: {cid}",
                "type": "bridge_card",
                "radius": 11.0,
                "syndicate_id": syns[0] if syns else "",
                "target_x": round(bx, 1),
                "target_y": round(by, 1),
                "details": {
                    "Card ID": cid,
                    "Type": "CROSS-CAMPAIGN PIVOT CARD",
                    "Syndicates Involved": ", ".join(syns),
                    "Merchants Targeted": ", ".join(merchs),
                    "Total Volume": f"${vol:,.2f}",
                    "Tx Count": tx_cnt,
                    "Approval Rate": f"{appr_pct}%",
                    "Latest Response": c_meta.get("latest_response", "N/A"),
                    "VAAI Score": c_meta.get("vaai_score", "N/A"),
                },
            })

        # Filter merchants to top targeted merchants + any merchant needed by bridge cards or unrolled cards
        merchants_needed = set()
        for cid in bridge_cards:
            merchants_needed.update(card_merchants.get(cid, set()))
        for cid in unrolled_card_ids:
            merchants_needed.update(card_merchants.get(cid, set()))

        top_merchants_by_volume = set(
            m for m, _ in sorted(merchant_tx_counts.items(), key=lambda x: x[1], reverse=True)[:self.max_merchants]
        )
        retained_merchants = top_merchants_by_volume | merchants_needed
        active_merchant_meta = {m: meta for m, meta in merchant_meta.items() if m in retained_merchants}

        # Register Merchant Nodes
        m_angle_step = 2.0 * math.pi / max(1, len(active_merchant_meta))
        for m_idx, (mid, meta) in enumerate(active_merchant_meta.items()):
            targeting_syns: Set[str] = set()
            for r in fraud_records:
                if str(r.get("merchant_id", "")) == mid:
                    s = str(r.get("syndicate_id", ""))
                    if s:
                        targeting_syns.add(s)

            if len(targeting_syns) > 1:
                # Merchants targeted by multiple threat groups spread along central corridor
                m_offset_y = (m_idx - len(active_merchant_meta) / 2.0) * 36.0
                mx = cx + (math.sin(m_idx * 1.6) * 90.0)
                my = cy + m_offset_y
            elif targeting_syns:
                target_syn = next(iter(targeting_syns))
                sax, say = syndicate_anchors.get(target_syn, (cx, cy))
                dir_x = 1.0 if sax < cx else -1.0
                mx = sax + dir_x * 230.0 + (math.cos(m_idx * 1.3) * 60.0)
                my = say + (math.sin(m_idx * 1.3) * 150.0)
            else:
                mx = cx + 240.0 * math.cos(m_idx * m_angle_step)
                my = cy + 180.0 * math.sin(m_idx * m_angle_step)

            register_node({
                "id": mid,
                "label": f"MID: {mid} (MCC {meta.get('mcc', '')})",
                "type": "merchant",
                "radius": 13.0,
                "target_x": round(mx, 1),
                "target_y": round(my, 1),
                "details": {
                    "Merchant ID": mid,
                    "MCC": meta.get("mcc", "N/A"),
                    "Category": meta.get("category", "N/A"),
                    "Channel": meta.get("channel", "N/A"),
                    "Attacking Syndicates": ", ".join(targeting_syns) if targeting_syns else "N/A",
                },
            })

        # Register Mule Nodes (Tiered Layering)
        for u_idx, (mule_id, meta) in enumerate(mule_meta.items()):
            sid = meta.get("syndicate_id") or (unique_syndicates[0] if unique_syndicates else "")
            sax, say = syndicate_anchors.get(sid, (cx, cy))
            tier_str = meta.get("tier", "TIER_1_SMURF")
            tier_dist_x = 270.0 if "TIER_1" in tier_str else (340.0 if "TIER_2" in tier_str else 420.0)
            tier_dist_y = 190.0 if "TIER_1" in tier_str else (250.0 if "TIER_2" in tier_str else 310.0)
            fan_dir = -1.0 if sax < cx else 1.0
            fan_angle = (u_idx * 0.7) - math.pi / 2.5
            ux = sax + fan_dir * tier_dist_x * (0.6 + 0.4 * abs(math.cos(fan_angle)))
            uy = say + tier_dist_y * math.sin(fan_angle)

            register_node({
                "id": mule_id,
                "label": f"Mule: {mule_id}",
                "type": "mule",
                "tier": tier_str,
                "radius": 12.0,
                "syndicate_id": sid,
                "target_x": round(ux, 1),
                "target_y": round(uy, 1),
                "details": {
                    "Beneficiary Mule": mule_id,
                    "FinCEN Layer": tier_str,
                    "Liquidation Channel": meta.get("channel", "N/A"),
                    "Affiliated Syndicate": sid,
                },
            })

        # 5. Weighted Edge Consolidation
        edge_accumulator: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        def record_edge(source: str, target: str, rel_type: str, amount: float, approved: bool) -> None:
            if source not in node_ids or target not in node_ids:
                return
            key = (source, target, rel_type)
            if key not in edge_accumulator:
                edge_accumulator[key] = {
                    "source": source,
                    "target": target,
                    "type": rel_type,
                    "count": 0,
                    "amount": 0.0,
                    "approved_count": 0,
                    "declined_count": 0,
                }
            edge_accumulator[key]["count"] += 1
            edge_accumulator[key]["amount"] += amount
            if approved:
                edge_accumulator[key]["approved_count"] += 1
            else:
                edge_accumulator[key]["declined_count"] += 1

        # Direct Syndicate -> Botnet operator links
        for bid, bmeta in botnet_meta.items():
            sid = bmeta.get("syndicate_id")
            if sid and sid in node_ids and bid in node_ids:
                record_edge(sid, bid, "OPERATES", 0.0, True)

        # Botnet -> Unrolled Cardholder links
        for cid in unrolled_card_ids:
            bots = card_botnets.get(cid, set())
            for bid in bots:
                if bid in node_ids and cid in node_ids:
                    record_edge(bid, cid, "COMPROMISES", card_volumes[cid], True)

        # Botnet -> Remainder Campaign links
        for (sid, bid, scen), cards in campaign_groups.items():
            remainder_cards = [c for c in cards if c not in unrolled_card_ids]
            if not remainder_cards:
                continue
            camp_id = f"CAMP_{bid}_{list(campaign_groups.keys()).index((sid, bid, scen)):02d}"
            tot_vol = sum(card_volumes[c] for c in remainder_cards)
            tot_tx = sum(card_tx_counts[c] for c in remainder_cards)
            tot_appr = sum(card_approvals[c] for c in remainder_cards)
            if bid in node_ids and camp_id in node_ids:
                key = (bid, camp_id, "CONTROLS")
                edge_accumulator[key] = {
                    "source": bid,
                    "target": camp_id,
                    "type": "CONTROLS",
                    "count": tot_tx,
                    "amount": round(tot_vol, 2),
                    "approved_count": tot_appr,
                    "declined_count": tot_tx - tot_appr,
                }

        # Process all fraud records for transactions into Merchants and Mules
        for r in fraud_records:
            cid = str(r.get("card_id", ""))
            mid = str(r.get("merchant_id", ""))
            mule_id = str(r.get("beneficiary_account_id", ""))
            bid = str(r.get("botnet_cluster_id", ""))
            amt = float(r.get("amount", 0.0))
            resp = str(r.get("response_code", "00"))
            is_appr = (resp in ("00", "10"))

            # Determine the source entity for the card's action:
            # 1. If bridge card, source is the card itself
            # 2. If unrolled cardholder, source is the card itself
            # 3. If rolled up into campaign remainder, source is the campaign node
            if cid in bridge_cards or cid in unrolled_card_ids:
                card_source = cid
            else:
                card_source = card_to_campaign_map.get(cid)

            # Link Botnet -> Bridge Card
            if cid in bridge_cards and bid and bid in node_ids:
                record_edge(bid, cid, "ATTACKS", amt, is_appr)

            # Link Card/Campaign -> Merchant
            if card_source and mid and mid in node_ids:
                record_edge(card_source, mid, "TRANSACTS", amt, is_appr)

            # Link Card/Campaign -> Mule
            if card_source and mule_id and mule_id in node_ids:
                record_edge(card_source, mule_id, "CASH_OUT", amt, is_appr)

        # Build final formatted link list with stroke width and curvature offsets
        links: List[Dict[str, Any]] = []
        for (src, dst, rel), data in edge_accumulator.items():
            cnt = data["count"]
            amt = round(data["amount"], 2)
            stroke_w = min(7.0, max(1.2, 1.2 + 1.5 * math.log10(max(1, cnt))))
            
            pair_hash = hash(f"{src}-{dst}")
            curvature = ((pair_hash % 5) - 2) * 12.0

            links.append({
                "source": src,
                "target": dst,
                "type": rel,
                "count": cnt,
                "amount": amt,
                "approved_count": data["approved_count"],
                "declined_count": data["declined_count"],
                "stroke_width": round(stroke_w, 2),
                "curvature": curvature,
            })

        # STRICT FORENSIC PRUNING:
        # Eliminate all zero-degree disconnected nodes so that no orphan/free-floating nodes appear on the canvas
        connected_node_ids = set()
        for l in links:
            connected_node_ids.add(l["source"])
            connected_node_ids.add(l["target"])

        nodes = [n for n in nodes if n["id"] in connected_node_ids]

        summary = {
            "syndicates_count": sum(1 for n in nodes if n["type"] == "syndicate"),
            "botnets_count": sum(1 for n in nodes if n["type"] == "botnet"),
            "campaigns_count": sum(1 for n in nodes if n["type"] == "breach_campaign"),
            "bridge_cards_count": sum(1 for n in nodes if n["type"] == "bridge_card"),
            "cardholders_count": sum(1 for n in nodes if n["type"] == "card"),
            "raw_cards_represented": len(card_metadata),
            "merchants_count": sum(1 for n in nodes if n["type"] == "merchant"),
            "mules_count": sum(1 for n in nodes if n["type"] == "mule"),
            "total_links": len(links),
        }

        return {
            "nodes": nodes,
            "links": links,
            "summary": summary,
        }

    def _empty_summary(self) -> Dict[str, Any]:
        return {
            "syndicates_count": 0,
            "botnets_count": 0,
            "campaigns_count": 0,
            "bridge_cards_count": 0,
            "raw_cards_represented": 0,
            "merchants_count": 0,
            "mules_count": 0,
            "total_links": 0,
        }
