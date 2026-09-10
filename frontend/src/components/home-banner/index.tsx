import decImg from '../../assets/home_banner_dec.png'
import styles from './index.module.scss'

export default function HomeBanner() {
  return (
    <div className={styles.banner}>
      <div className={styles.content}>
        <h1 className={styles.title}>Traceable RAG</h1>
        <p className={styles.subtitle}>
          Source-grounded answers with visible retrieval progress
        </p>
      </div>
      <div className={styles.decoration}>
        <img src={decImg} alt="" className={styles.decImage} />
      </div>
    </div>
  )
}
