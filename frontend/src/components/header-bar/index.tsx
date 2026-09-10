import styles from './index.module.scss'

export default function HeaderBar(props: { className?: string }) {
  return (
    <header className={`${styles['header-bar']} ${props.className || ''}`} />
  )
}
