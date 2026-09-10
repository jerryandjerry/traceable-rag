import { Tag } from 'antd'
import Color from 'color'
import { useMemo } from 'react'

const map = {
  unparsed: {
    text: 'Unparsed',
    color: '#67C23A',
  },
  cancel: {
    text: 'Cancelled',
    color: '#E6A23C',
  },
  success: {
    text: 'Completed',
    color: 'var(--ant-color-primary)',
  },
  failed: {
    text: 'Failed',
    color: '#F56C6C',
  },
}

export function Status(props: { status: keyof typeof map }) {
  const { status } = props
  const { text, color } = useMemo(() => {
    return (
      map[status] ?? {
        color: '#999',
        text: status,
      }
    )
  }, [status])

  const backgroundColor = useMemo(() => {
    return new Color(color).alpha(0.1).toString()
  }, [color])

  const borderColor = useMemo(() => {
    return new Color(color).alpha(0.3).toString()
  }, [color])

  return <Tag style={{ borderColor, color, backgroundColor }}>{text}</Tag>
}
