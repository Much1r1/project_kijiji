// src/hooks/useLiveBGP.js
//
// Connects to the FastAPI bridge WebSocket and streams live BGP events
// into the dashboard. Falls back to synthetic data if bridge is unreachable.
//
// Usage in App.jsx:
//   import { useLiveBGP } from './hooks/useLiveBGP'
//   const { events, trombones, stats, isLive } = useLiveBGP()

import { useState, useEffect, useRef, useCallback } from "react"

const BRIDGE_WS  = import.meta.env.VITE_BRIDGE_WS  || "ws://localhost:8000/ws/events"
const MAX_EVENTS = 100   // keep last N events in state
const MAX_TROMBONES = 50

// Synthetic fallback data — same structure as real events
// Used when bridge is not reachable (dev without Python running)
const SYNTHETIC_NODES = ["NBO", "LOS", "JNB", "CPT", "ACC", "DAR", "ADD", "KIN", "KMP", "LUS"]
const SYNTHETIC_HUBS  = ["London", "Paris", "Frankfurt", "Amsterdam", "New York"]

function makeSyntheticEvent() {
  const src = SYNTHETIC_NODES[Math.floor(Math.random() * SYNTHETIC_NODES.length)]
  let dst = src
  while (dst === src) dst = SYNTHETIC_NODES[Math.floor(Math.random() * SYNTHETIC_NODES.length)]

  const isTrombone = Math.random() < 0.35
  const via        = SYNTHETIC_HUBS[Math.floor(Math.random() * SYNTHETIC_HUBS.length)]
  const ratio      = isTrombone ? +(2.1 + Math.random() * 3.5).toFixed(2) : null
  const wasted     = isTrombone ? Math.round((ratio - 1) * 50 + 30) : null

  return {
    ts:                   new Date().toISOString(),
    src_node:             src,
    dst_node:             dst,
    src_op:               `Synthetic-${src}`,
    collector:            "synthetic",
    prefix:               `41.${Math.floor(Math.random()*255)}.${Math.floor(Math.random()*255)}.0/24`,
    path_len:             3 + Math.floor(Math.random() * 5),
    is_trombone:          isTrombone,
    trombone_via:         isTrombone ? via : null,
    trombone_ratio:       ratio,
    trombone_wasted_ms:   wasted,
    direct_km:            isTrombone ? Math.round(1000 + Math.random() * 4000) : null,
    actual_km:            isTrombone ? Math.round((1000 + Math.random() * 4000) * ratio) : null,
    _synthetic:           true,
  }
}


export function useLiveBGP() {
  const [events,    setEvents]    = useState([])
  const [trombones, setTrombones] = useState([])
  const [stats,     setStats]     = useState({ total: 0, tromboneCount: 0 })
  const [isLive,    setIsLive]    = useState(false)
  const [status,    setStatus]    = useState("connecting") // connecting | live | synthetic | error

  const wsRef       = useRef(null)
  const synthTimer  = useRef(null)
  const reconnTimer = useRef(null)

  const pushEvent = useCallback((evt) => {
    setEvents(prev => {
      const next = [evt, ...prev]
      return next.length > MAX_EVENTS ? next.slice(0, MAX_EVENTS) : next
    })
    if (evt.is_trombone) {
      setTrombones(prev => {
        const next = [evt, ...prev]
        return next.length > MAX_TROMBONES ? next.slice(0, MAX_TROMBONES) : next
      })
    }
    setStats(prev => ({
      total:          prev.total + 1,
      tromboneCount:  prev.tromboneCount + (evt.is_trombone ? 1 : 0),
    }))
  }, [])

  // ── Synthetic fallback ──────────────────────────────────────────────────
  const startSynthetic = useCallback(() => {
    setStatus("synthetic")
    setIsLive(false)
    if (synthTimer.current) return  // already running
    synthTimer.current = setInterval(() => {
      const batch = Math.floor(1 + Math.random() * 3)
      for (let i = 0; i < batch; i++) pushEvent(makeSyntheticEvent())
    }, 2500)
  }, [pushEvent])

  const stopSynthetic = useCallback(() => {
    if (synthTimer.current) {
      clearInterval(synthTimer.current)
      synthTimer.current = null
    }
  }, [])

  // ── WebSocket connection ────────────────────────────────────────────────
  const connect = useCallback(() => {
    // Don't try to connect if we're already open
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(BRIDGE_WS)
      wsRef.current = ws

      ws.onopen = () => {
        setStatus("live")
        setIsLive(true)
        stopSynthetic()
        if (reconnTimer.current) {
          clearTimeout(reconnTimer.current)
          reconnTimer.current = null
        }
        console.log("[useLiveBGP] Connected to bridge:", BRIDGE_WS)
      }

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)

          // Initial connection message includes history
          if (msg.type === "connected") {
            if (msg.history?.length) {
              msg.history.reverse().forEach(pushEvent)
            }
            if (msg.stats) {
              setStats({
                total:         msg.stats.total_events || 0,
                tromboneCount: msg.stats.trombone_events || 0,
              })
            }
            return
          }

          // Regular event
          pushEvent(msg)
        } catch {
          // ignore malformed messages
        }
      }

      ws.onerror = () => {
        console.warn("[useLiveBGP] WebSocket error — falling back to synthetic")
        setStatus("error")
      }

      ws.onclose = () => {
        if (status === "live") {
          console.warn("[useLiveBGP] Bridge disconnected — starting synthetic fallback")
        }
        startSynthetic()
        // Attempt reconnect every 15s
        if (!reconnTimer.current) {
          reconnTimer.current = setTimeout(() => {
            reconnTimer.current = null
            connect()
          }, 15_000)
        }
      }
    } catch (err) {
      console.warn("[useLiveBGP] Cannot connect to bridge:", err.message)
      startSynthetic()
    }
  }, [pushEvent, startSynthetic, stopSynthetic, status])

  // ── Lifecycle ───────────────────────────────────────────────────────────
  useEffect(() => {
    connect()
    return () => {
      wsRef.current?.close()
      stopSynthetic()
      if (reconnTimer.current) clearTimeout(reconnTimer.current)
    }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  return { events, trombones, stats, isLive, status }
}