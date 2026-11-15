
import PrecipitationChart from './components/PrecipitationChart'

function App() {
  return (
    <main style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '40px 24px',
      textAlign: 'left'
    }}>
      <header style={{ marginBottom: '32px' }}>
        <p style={{ color: '#61e0c0' }}>
          IONA / EARTH INTELLIGENCE
        </p>

        <h1>Descubra o que as médias escondem.</h1>

        <p>
          Plataforma de investigação ambiental utilizando
          dados científicos da NASA.
        </p>
      </header>

      <PrecipitationChart />
    </main>
  )
}

export default App
