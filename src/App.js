import React from 'react';
import { ChakraProvider, Box, Heading, Link, VStack, Button } from '@chakra-ui/react';
import { BrowserRouter as Router, Route, Routes, Link as RouterLink } from 'react-router-dom';
import logo from './logo.svg';
import './App.css';
import EducationalModule from './components/EducationalModule';

function App() {
  return (
    <ChakraProvider>
      <Router>
        <Box className="App">
          <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
            <Heading as="h1" size="xl" mb={4}>
              Oxygen Agent
            </Heading>
            <VStack spacing={4}>
              <Button as={RouterLink} to="/educational" colorScheme="teal" size="lg">
                Educational Module
              </Button>
            </VStack>
            <p>
              Edit <code>src/App.js</code> and save to reload.
            </p>
            <Link
              className="App-link"
              href="https://reactjs.org"
              isExternal
            >
              Learn React
            </Link>
          </header>
          <Routes>
            <Route path="/educational" element={<EducationalModule />} />
          </Routes>
        </Box>
      </Router>
    </ChakraProvider>
  );
}

export default App;
